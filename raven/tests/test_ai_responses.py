import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import frappe
import httpx2
from agents import Agent, OpenAIProvider, RunConfig, Runner
from frappe.tests import IntegrationTestCase
from openai import AsyncOpenAI

from raven.ai.agents_integration import RavenAgentManager, get_model_settings
from raven.ai.ai import handle_ai_thread_message, handle_bot_dm
from raven.ai.openai_client import is_reasoning_model
from raven.raven_bot.doctype.raven_bot.raven_bot import RavenBot

BOT_CLIENT = "raven.raven_bot.doctype.raven_bot.raven_bot.get_open_ai_client"


def sent_to_responses_api(model: str, model_settings) -> dict:
	"""Run an agent against a mocked /v1/responses and return the request body Raven sent."""
	body = {}

	def respond(request):
		body.update(json.loads(request.content))
		message = {
			"type": "message",
			"id": "msg_test",
			"status": "completed",
			"role": "assistant",
			"content": [{"type": "output_text", "text": "ok", "annotations": []}],
		}
		return httpx2.Response(
			200,
			json={
				"id": "resp_test",
				"object": "response",
				"created_at": 0,
				"status": "completed",
				"model": model,
				"output": [message],
				"parallel_tool_calls": True,
				"tool_choice": "auto",
				"tools": [],
			},
		)

	client = AsyncOpenAI(
		api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
	)
	provider = OpenAIProvider(openai_client=client, use_responses=True)
	agent = Agent(
		name="Test", instructions="Test", model=provider.get_model(model), model_settings=model_settings
	)
	Runner.run_sync(agent, "hello", run_config=RunConfig(tracing_disabled=True))
	return body


def make_bot(**fields):
	defaults = {"model": "gpt-4o", "temperature": 1, "top_p": 1, "reasoning_effort": None}
	return SimpleNamespace(**{**defaults, **fields})


class TestAIResponses(IntegrationTestCase):
	def tearDown(self):
		frappe.db.rollback()

	def test_reasoning_model_names(self):
		for model in ["o1", "o3-mini", "o4-mini", "gpt-5", "gpt-5.6-terra", "gpt-6-astra"]:
			self.assertTrue(is_reasoning_model(model), model)

		for model in ["gpt-4o", "gpt-4.1", "gpt-oss-20b", "chatgpt-4o-latest", "llama3", "", None]:
			self.assertFalse(is_reasoning_model(model), model)

	def test_bots_with_a_stale_assistant_id_use_the_agents_sdk(self):
		bot = make_bot(openai_assistant_id="asst_retired", model_provider="OpenAI")
		channel = SimpleNamespace(thread_bot="Test Bot", openai_thread_id="thread_retired")

		with patch("raven.ai.ai.handle_bot_dm_with_agents") as dm_with_agents:
			handle_bot_dm("message", bot)
		dm_with_agents.assert_called_once_with("message", bot)

		with (
			patch("raven.ai.ai.frappe.get_cached_doc", return_value=bot),
			patch("raven.ai.ai.handle_ai_thread_message_with_agents") as thread_with_agents,
		):
			handle_ai_thread_message("message", channel)
		thread_with_agents.assert_called_once_with("message", channel, bot)

	def test_reasoning_models_get_effort_and_no_sampling_params(self):
		for model in ["gpt-5.6-terra", "gpt-6-astra", "o3"]:
			bot = make_bot(model=model, reasoning_effort="high")
			body = sent_to_responses_api(model, get_model_settings(bot))

			self.assertEqual(body["reasoning"], {"effort": "high"}, model)
			self.assertNotIn("temperature", body, model)
			self.assertNotIn("top_p", body, model)

	def test_other_models_keep_sampling_params(self):
		bot = make_bot(model="gpt-4o", temperature=0.2, top_p=0.9, reasoning_effort="high")
		body = sent_to_responses_api("gpt-4o", get_model_settings(bot))

		self.assertEqual(body["temperature"], 0.2)
		self.assertEqual(body["top_p"], 0.9)
		self.assertNotIn("reasoning", body)

	def test_saving_a_function_used_by_a_bot_makes_no_openai_call(self):
		function = frappe.get_doc(
			{
				"doctype": "Raven AI Function",
				"function_name": "fetch_test_user",
				"description": "Fetch a user",
				"type": "Get Document",
				"reference_doctype": "User",
			}
		).insert()
		self.make_saved_bot(bot_functions=[{"function": function.name}])

		with patch("raven.ai.openai_client.OpenAI", side_effect=AssertionError("OpenAI was called")):
			function.description = "Fetch a user by email"
			function.save()

	def test_file_sources_reach_the_vector_store_on_save(self):
		bot = self.make_saved_bot()
		client = MagicMock()
		client.vector_stores.create.return_value = SimpleNamespace(id="vs_test")

		with (
			patch(BOT_CLIENT, return_value=client),
			patch.object(RavenBot, "get_openai_file_ids", return_value=["file_a"]),
		):
			bot.enable_file_search = 1
			bot.save()

		client.vector_stores.create.assert_called_once_with(name=bot.name, file_ids=["file_a"])
		self.assertEqual(frappe.db.get_value("Raven Bot", bot.name, "openai_vector_store_id"), "vs_test")

		client.vector_stores.files.list.return_value = SimpleNamespace(
			data=[SimpleNamespace(id="file_a")]
		)
		with (
			patch(BOT_CLIENT, return_value=client),
			patch.object(RavenBot, "get_openai_file_ids", return_value=["file_a", "file_b"]),
		):
			bot.reload()
			bot.save()

		client.vector_stores.file_batches.create.assert_called_once_with(
			vector_store_id="vs_test", file_ids=["file_b"]
		)

	def test_file_search_creates_the_missing_vector_store_on_first_use(self):
		bot = SimpleNamespace(openai_vector_store_id=None)
		bot.sync_vector_store = lambda: setattr(bot, "openai_vector_store_id", "vs_new")
		manager = RavenAgentManager.__new__(RavenAgentManager)
		manager.bot_doc = bot

		tool = manager._create_file_search_tool()

		self.assertEqual(tool.vector_store_ids, ["vs_new"])

	def test_code_interpreter_gets_the_bot_file_sources(self):
		manager = RavenAgentManager.__new__(RavenAgentManager)
		manager.file_handler = None
		manager.bot_doc = SimpleNamespace(
			model_provider="OpenAI",
			enable_file_search=0,
			enable_code_interpreter=1,
			bot_functions=[],
			get_openai_file_ids=lambda file_types: ["file_csv"],
		)

		with patch("raven.ai.sdk_tools.create_raven_tools", return_value=[]):
			manager._setup_tools()

		self.assertEqual(manager.tools[0].tool_config["container"]["file_ids"], ["file_csv"])

	def make_saved_bot(self, **fields):
		return frappe.get_doc(
			{
				"doctype": "Raven Bot",
				"bot_name": "Responses Test Bot",
				"is_ai_bot": 1,
				"model_provider": "OpenAI",
				"model": "gpt-4o",
				"instruction": "Answer briefly.",
				**fields,
			}
		).insert()

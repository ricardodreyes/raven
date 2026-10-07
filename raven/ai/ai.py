import frappe

# Import agents integration - no fallback needed
from raven.ai.agents_integration import handle_ai_request_sync
from raven.ai.google_ai import run_document_ai_processor


def handle_bot_dm(message, bot):
	"""
	Function to handle direct messages to the bot.

	Every bot runs on the Agents SDK. OpenAI shut down the Assistants API, so a leftover
	openai_assistant_id on older bots is ignored.
	"""
	return handle_bot_dm_with_agents(message, bot)


def handle_bot_dm_with_agents(message, bot):
	"""
	Handle direct messages using Agents SDK.
	"""

	# If the message is a poll, send a message to the user that we don't support polls for AI yet
	if message.message_type == "Poll":
		bot.send_message(
			channel_id=message.channel_id,
			text="Sorry, I don't support polls yet. Please send a text message or file.",
		)
		return

	# Create thread channel for the conversation
	thread_channel = frappe.get_doc(
		{
			"doctype": "Raven Channel",
			"channel_name": message.name,
			"type": "Private",
			"is_thread": 1,
			"is_ai_thread": 1,
			"is_dm_thread": 1,
			"thread_bot": bot.name,
			# No more openai_thread_id for Agents SDK
		}
	).insert()

	# Update the message to mark it as a thread
	message.is_thread = 1
	message.save()
	# We need to commit here since the response will be processed asynchronously
	# Manual commit required: AI processing happens in background job that needs the message to exist in DB
	frappe.db.commit()  # nosemgrep

	# Send initial thinking message
	frappe.publish_realtime(
		"ai_event",
		{
			"text": "Raven AI is thinking...",
			"channel_id": thread_channel.name,
			"bot": bot.name,
		},
		doctype="Raven Channel",
		docname=thread_channel.name,
		after_commit=True,
	)

	# Send event to automatically open the thread
	publish_ai_thread_created_event(message, message.channel_id)

	# Process message with Agents SDK
	process_message_with_agent(
		message=message, bot=bot, channel_id=thread_channel.name, is_new_conversation=True
	)


def handle_ai_thread_message(message, channel):
	"""
	Function to handle messages in an AI thread.

	History is rebuilt from the thread's Raven Messages, so threads started on the
	Assistants API carry on without their openai_thread_id.
	"""

	bot = frappe.get_cached_doc("Raven Bot", channel.thread_bot)
	return handle_ai_thread_message_with_agents(message, channel, bot)


def handle_ai_thread_message_with_agents(message, channel, bot):
	"""
	Handle thread messages using Agents SDK.
	"""

	# Skip file/image messages without text - they'll be handled when the user sends a follow-up
	if message.message_type in ["File", "Image"] and not message.text and not message.content:
		frappe.log_error(
			f"Skipping file-only message in AI thread: {message.file}", "AI Thread File Skip"
		)
		return

	# Send thinking message
	frappe.publish_realtime(
		"ai_event",
		{
			"text": "Raven AI is thinking...",
			"channel_id": channel.name,
			"bot": bot.name,
		},
		doctype="Raven Channel",
		docname=channel.name,
	)

	# Process message with Agents SDK
	process_message_with_agent(
		message=message, bot=bot, channel_id=channel.name, is_new_conversation=False, channel=channel
	)


def extract_file_content_for_agent(file_url: str, file_extension: str, bot, file_handler):
	"""
	Extract content from a file for agent processing.

	Tries Google Document AI first if enabled, falls back to basic extraction.
	"""
	extracted_content = ""

	# Try Google Document AI first if enabled
	if hasattr(bot, "use_google_document_parser") and bot.use_google_document_parser:
		if hasattr(bot, "google_document_processor_id") and bot.google_document_processor_id:
			if file_extension in ["jpg", "jpeg", "png", "pdf"]:
				try:
					extracted_content = run_document_ai_processor(
						bot.google_document_processor_id, file_url, file_extension
					)
					if extracted_content:
						return extracted_content
				except Exception as e:
					frappe.log_error(
						f"Error extracting content with Google Document AI: {str(e)}\nFile: {file_url}",
						"Google Document AI Error",
					)

	# Fallback to basic extraction
	if not extracted_content:
		try:
			file_doc = frappe.get_doc("File", {"file_url": file_url})
			file_path = file_doc.get_full_path()

			if file_extension == "pdf":
				extracted_content = file_handler._extract_pdf_content(file_path)
			elif file_extension in ["txt", "md", "json"]:
				with open(file_path, encoding="utf-8") as f:
					extracted_content = f.read()
			elif file_extension in ["xlsx", "xls", "csv"]:
				extracted_content = file_handler._convert_spreadsheet_to_markdown(file_path, file_extension)
			# For images, return empty - Google AI should handle them

		except Exception as e:
			frappe.log_error(
				f"Error extracting file content: {str(e)}\nFile: {file_url}", "File Extraction Error"
			)

	return extracted_content


def process_message_with_agent(
	message, bot, channel_id: str, is_new_conversation: bool, channel=None
):
	"""
	Process a message using the Agents SDK.

	This function handles both new conversations and existing threads.
	"""

	# Track files in conversation
	from raven.ai.conversation_file_handler import ConversationFileHandler

	file_handler = ConversationFileHandler(channel_id=channel_id, bot=bot)

	# Check if this is a text message following a recent file upload
	# to combine them into a single AI request
	recent_file_message = None
	if message.message_type == "Text" and channel:
		# Look for a file message from the same user in the last 30 seconds
		from datetime import datetime, timedelta

		cutoff_time = datetime.now() - timedelta(seconds=30)

		recent_messages = frappe.get_all(
			"Raven Message",
			filters={
				"channel_id": channel.name,
				"owner": message.owner,
				"message_type": ["in", ["File", "Image"]],
				"creation": [">", cutoff_time],
				"is_bot_message": 0,
			},
			fields=["name", "file", "message_type", "text", "content"],
			order_by="creation desc",
			limit=1,
		)

		if recent_messages:
			recent_file = recent_messages[0]
			# Check if the file message had no text
			if not recent_file.text and not recent_file.content:
				recent_file_message = recent_file
				# Add this file to the file handler
				file_handler.add_conversation_file(recent_file)

	# Prepare the message content
	if message.message_type in ["File", "Image"]:
		# Add file to conversation context
		file_handler.add_conversation_file(message)

		# If it's just a file upload without any text, don't process it yet
		# Wait for the user to ask a question about it
		if not message.text and not message.content:
			return {"success": True, "response": None}

		# Extract file content immediately (like Assistants API does)
		if "fid" in message.file:
			file_url = message.file.split("?fid=")[0]
		else:
			file_url = message.file

		# Get file extension for processing
		file_extension = file_url.split(".")[-1].lower() if "." in file_url else ""

		# Extract file content using Google AI if enabled, or fallback to basic extraction
		extracted_content = extract_file_content_for_agent(file_url, file_extension, bot, file_handler)

		# Build message content with extracted text
		if message.message_type == "File":
			content = f"[User uploaded a file: {file_url}]"
			if extracted_content:
				content += f"\n\nExtracted content from the file:\n{extracted_content}"
			if message.text or message.content:
				content += f"\n\nUser's question: {message.text or message.content}"
		else:
			content = f"[User uploaded an image: {file_url}]"
			if extracted_content:
				content += f"\n\nExtracted content from the image:\n{extracted_content}"
			if message.text or message.content:
				content += f"\n\nUser's question: {message.text or message.content}"
	else:
		content = message.text or message.content or ""

		# If we found a recent file upload, extract and prepend its content
		if recent_file_message:
			file_url = recent_file_message.file
			if "fid" in file_url:
				file_url = file_url.split("?fid=")[0]

			# Extract content from recent file
			file_extension = file_url.split(".")[-1].lower() if "." in file_url else ""
			extracted_content = extract_file_content_for_agent(file_url, file_extension, bot, file_handler)

			file_prefix = f"[User uploaded a {'file' if recent_file_message.message_type == 'File' else 'image'}: {file_url}]"
			if extracted_content:
				file_prefix += f"\n\nExtracted content from the file:\n{extracted_content}\n\n"
			else:
				file_prefix += "\n"

			content = file_prefix + content

	# Get conversation history if this is an existing thread
	conversation_history = []
	if not is_new_conversation and channel:
		# Fetch previous messages from the channel
		messages = frappe.get_all(
			"Raven Message",
			filters={"channel_id": channel.name},
			fields=[
				"text",
				"content",
				"owner",
				"creation",
				"bot",
				"message_type",
				"file",
				"is_bot_message",
			],
			order_by="creation asc",
			limit=20,  # Limit to last 20 messages for context
		)

		for msg in messages[:-1]:  # Exclude the current message
			# Use text field which contains the actual message content
			msg_text = msg.text or msg.content or ""

			if msg.bot or msg.is_bot_message:
				conversation_history.append({"role": "assistant", "content": msg_text})
			else:
				if msg.message_type in ["File", "Image"]:
					# DON'T add historical files - only the current message file should be analyzed
					# Just add a reference to the file in conversation history

					file_url = msg.file.split("?fid=")[0] if "fid" in msg.file else msg.file
					msg_content = (
						f"[User uploaded a {'file' if msg.message_type == 'File' else 'image'}: {file_url}]"
					)
					if msg_text:
						msg_content += f"\n{msg_text}"
					conversation_history.append({"role": "user", "content": msg_content})
				else:
					conversation_history.append({"role": "user", "content": msg_text})

	# Use the improved sync handler
	try:
		# Use Agents SDK for both OpenAI and Local LLM
		response = handle_ai_request_sync(
			bot=bot,
			message=content,
			channel_id=channel_id,
			conversation_history=conversation_history,
			file_handler=file_handler,
		)

		if response["success"]:
			# Only send a response if there is one
			if response["response"] is not None:
				bot.send_message(channel_id=channel_id, text=response["response"])
			# If response is None (e.g., file-only upload), don't send anything
		else:
			# Send error message
			error_text = "Sorry, I encountered an error while processing your request."
			if bot.debug_mode and response.get("error"):
				error_text += f"\n\nError: {response['error']}"

			bot.send_message(channel_id=channel_id, text=error_text)

		# Clear the "thinking" message after sending the response
		frappe.publish_realtime(
			"ai_event_clear",
			{
				"channel_id": channel_id,
			},
			doctype="Raven Channel",
			docname=channel_id,
			after_commit=True,
		)
	except Exception as e:
		import traceback

		frappe.log_error(
			f"Error calling handle_ai_request_sync: {str(e)}\n\nTraceback:\n{traceback.format_exc()}",
			"Raven AI",
		)
		# Send error message
		error_text = "I encountered an error while processing your request."
		if bot.debug_mode:
			error_text += f"\n\nError: {str(e)}"

		bot.send_message(channel_id=channel_id, text=error_text)

		# Clear the "thinking" message even on error
		frappe.publish_realtime(
			"ai_event_clear",
			{
				"channel_id": channel_id,
			},
			doctype="Raven Channel",
			docname=channel_id,
			after_commit=True,
		)


def publish_ai_thread_created_event(message, channel_id):
	"""
	Publish an event when an AI thread is created for auto-opening in frontend
	"""
	frappe.publish_realtime(
		"ai_thread_created",
		{"thread_id": message.name, "channel_id": channel_id, "is_ai_thread": True},
		user=message.owner,
		after_commit=False,
	)

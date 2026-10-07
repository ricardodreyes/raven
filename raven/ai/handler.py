import frappe


def get_variables_for_instructions():
	user = frappe.get_cached_doc("User", frappe.session.user)

	employee_company = None
	company = None
	employee_id = None
	department = None

	if "erpnext" in frappe.get_installed_apps():
		employee_id = frappe.db.exists("Employee", {"user_id": frappe.session.user})

		if employee_id:
			employee = frappe.get_cached_doc("Employee", employee_id)
			employee_company = employee.company
			department = employee.department

		company = frappe.defaults.get_user_default("company") or frappe.db.get_single_value(
			"Global Defaults", "default_company"
		)

	return {
		"first_name": user.first_name,
		"full_name": user.full_name,
		"email": user.email,
		"user_id": frappe.session.user,
		"company": company,
		"employee_id": employee_id,
		"department": department,
		"employee_company": employee_company,
		"lang": user.language.upper() if user.language else "EN",
	}

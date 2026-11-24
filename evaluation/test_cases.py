T01 = {
    "id": "T01",
    "question": "What is the monthly membership charge?",
    "expected_tool": "search_invoices",
    "expected_evidence": "USD $37.50",
    "expected_source": "sample.pdf",
    "expected_amount": "37.50",
    "expected_description": "Premium Membership - Monthly",
}

T02 = {
    "id": "T02",
    "question": "What is the invoice number?",
    "expected_tool": "search_invoices",
    "expected_evidence": "Invoice #13594027",
    "expected_source": "sample.pdf",
    "expected_invoice_number": "13594027",
    "expected_description": "Invoice #",
}

T05 = {
    "id": "T05",
    "question": "Was invoice 13594027 paid?",
    "expected_tool": "search_invoices",
    "expected_evidence": "13594027",
    "should_abstain": True,
}

T06 = {
    "id": "T06",
    "question": "What is the weather in London?",
    "category": "out_of_scope",
    "should_abstain": True,
    "requires_tool": False,
}

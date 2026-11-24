"""Manually reviewed membership-charge labels and full source snapshots.

Stored chunk IDs are the evaluation units, including duplicate text.
Labels were assigned from full text, independently of retrieval distances.
"""

QUESTION = "How much was charged for the monthly membership?"
RELEVANCE_RULE = "A relevant chunk establishes monthly membership context and the charged amount USD $37.50."

GROUND_TRUTH: dict[str, int] = {'sample-p0-c0': 1,
 'sample-p0-c1': 1,
 'sample-p1-c0': 1,
 'sample-p1-c1': 1,
 'sample-p2-c0': 1,
 'sample-p2-c1': 1,
 'sample-p3-c0': 1,
 'sample-p3-c1': 1}

REVIEWED_CHUNKS = {'sample-p0-c0': {'page': 0,
                  'source': 'sample.pdf',
                  'text': 'Invoice #13594027\n'
                          'Description Total\n'
                          'Sub Total\n'
                          'Credit USD $0.00\n'
                          'Total\n'
                          'Transactions\n'
                          'Transaction Date Transaction ID Amount\n'
                          'Balance\n'
                          'PDF Generated on 06/23/2018\n'
                          'Powered by TCPDF (www.tcpdf.org)\n'
                          '1 Main St.\n'
                          'Bevery Hills, CA 90210\n'
                          'Phone: +1 (800) 555-5555\n'
                          'email@yourdomain.com\n'
                          ' \n'
                          ' \n'
                          '     USD $50.00        \n'
                          'Promotional Code: 25OFF - 25.00% Recurring Discount\n'
                          'USD $37.50\n'
                          'USD $37.50\n'
                          'USD $-12.50\n'
                          'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)'},
 'sample-p0-c1': {'page': 0,
                  'source': 'sample.pdf',
                  'text': 'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)\n'
                          'Your Business Name\n'
                          ' \n'
                          'Invoiced To\n'
                          'John Smith \n'
                          '123 Oak Ave.\n'
                          '   Los Angeles, CA 90017\n'
                          'ch_9CdwKtEDJAI2k4ZEVqkl0xQr06/23/2018 USD $37.50'},
 'sample-p1-c0': {'page': 1,
                  'source': 'sample.pdf',
                  'text': 'Invoice #13594027\n'
                          'Description Total\n'
                          'Sub Total\n'
                          'Credit USD $0.00\n'
                          'Total\n'
                          'Transactions\n'
                          'Transaction Date Transaction ID Amount\n'
                          'Balance\n'
                          'PDF Generated on 06/23/2018\n'
                          'Powered by TCPDF (www.tcpdf.org)\n'
                          '1 Main St.\n'
                          'Bevery Hills, CA 90210\n'
                          'Phone: +1 (800) 555-5555\n'
                          'email@yourdomain.com\n'
                          ' \n'
                          ' \n'
                          '     USD $50.00        \n'
                          'Promotional Code: 25OFF - 25.00% Recurring Discount\n'
                          'USD $37.50\n'
                          'USD $37.50\n'
                          'USD $-12.50\n'
                          'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)'},
 'sample-p1-c1': {'page': 1,
                  'source': 'sample.pdf',
                  'text': 'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)\n'
                          'Your Business Name\n'
                          ' \n'
                          'Invoiced To\n'
                          'John Smith \n'
                          '123 Oak Ave.\n'
                          '   Los Angeles, CA 90017\n'
                          'ch_9CdwKtEDJAI2k4ZEVqkl0xQr06/23/2018 USD $37.50'},
 'sample-p2-c0': {'page': 2,
                  'source': 'sample.pdf',
                  'text': 'Invoice #13594027\n'
                          'Description Total\n'
                          'Sub Total\n'
                          'Credit USD $0.00\n'
                          'Total\n'
                          'Transactions\n'
                          'Transaction Date Transaction ID Amount\n'
                          'Balance\n'
                          'PDF Generated on 06/23/2018\n'
                          'Powered by TCPDF (www.tcpdf.org)\n'
                          '1 Main St.\n'
                          'Bevery Hills, CA 90210\n'
                          'Phone: +1 (800) 555-5555\n'
                          'email@yourdomain.com\n'
                          ' \n'
                          ' \n'
                          '     USD $50.00        \n'
                          'Promotional Code: 25OFF - 25.00% Recurring Discount\n'
                          'USD $37.50\n'
                          'USD $37.50\n'
                          'USD $-12.50\n'
                          'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)'},
 'sample-p2-c1': {'page': 2,
                  'source': 'sample.pdf',
                  'text': 'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)\n'
                          'Your Business Name\n'
                          ' \n'
                          'Invoiced To\n'
                          'John Smith \n'
                          '123 Oak Ave.\n'
                          '   Los Angeles, CA 90017\n'
                          'ch_9CdwKtEDJAI2k4ZEVqkl0xQr06/23/2018 USD $37.50'},
 'sample-p3-c0': {'page': 3,
                  'source': 'sample.pdf',
                  'text': 'Invoice #13594027\n'
                          'Description Total\n'
                          'Sub Total\n'
                          'Credit USD $0.00\n'
                          'Total\n'
                          'Transactions\n'
                          'Transaction Date Transaction ID Amount\n'
                          'Balance\n'
                          'PDF Generated on 06/23/2018\n'
                          'Powered by TCPDF (www.tcpdf.org)\n'
                          '1 Main St.\n'
                          'Bevery Hills, CA 90210\n'
                          'Phone: +1 (800) 555-5555\n'
                          'email@yourdomain.com\n'
                          ' \n'
                          ' \n'
                          '     USD $50.00        \n'
                          'Promotional Code: 25OFF - 25.00% Recurring Discount\n'
                          'USD $37.50\n'
                          'USD $37.50\n'
                          'USD $-12.50\n'
                          'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)'},
 'sample-p3-c1': {'page': 3,
                  'source': 'sample.pdf',
                  'text': 'USD $37.50\n'
                          'Date: 06/23/2018\n'
                          'Premium Membership - Monthly - yourdomain.com (06/23/2018 - '
                          '07/23/2018)\n'
                          'Your Business Name\n'
                          ' \n'
                          'Invoiced To\n'
                          'John Smith \n'
                          '123 Oak Ave.\n'
                          '   Los Angeles, CA 90017\n'
                          'ch_9CdwKtEDJAI2k4ZEVqkl0xQr06/23/2018 USD $37.50'}}

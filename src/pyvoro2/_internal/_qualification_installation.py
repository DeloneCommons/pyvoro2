"""Unqualified source-tree installation.

Only the controlled external finalizer replaces this module in a completed
installation. RECORD_SHA256 authenticates canonical detached record bytes;
INSTALLATION_ID is separately bound by that record to avoid a recursive hash.
An ordinary source build or adjacent hand-written JSON cannot issue an anchor.
"""

RECORD_FILENAME = 'native_qualification_record.json'
RECORD_SHA256 = None
INSTALLATION_ID = None

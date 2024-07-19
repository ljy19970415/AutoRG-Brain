# -*- coding: utf-8 -*-

import re
from petrel_client.common.exception import InvalidS3UriError
# (?:...)
# A non-capturing version of regular parentheses. Matches whatever regular expression is inside the parentheses, but the substring matched by the group cannot be retrieved after performing a match or referenced later in the pattern.

# *?, +?, ??
# The '*', '+', and '?' qualifiers are all greedy; they match as much text as possible. Sometimes this behaviour isn’t desired; if the RE <.*> is matched against <a> b <c>, it will match the entire string, and not just <a>. Adding ? after the qualifier makes it perform the match in non-greedy or minimal fashion; as few characters as possible will be matched. Using the RE <.*?> will match only <a>.

# re.I
# re.IGNORECASE
# Perform case-insensitive matching; expressions like [A-Z] will match lowercase letters, too. This is not affected by the current locale. To get this effect on non-ASCII Unicode characters such as ü and Ü, add the UNICODE flag.
PATTERN = re.compile(r'^(?:([^:]+):)?s3://([^/]+)/(.+?)/?$', re.I)


def parse_s3_uri(uri):
    m = PATTERN.match(uri)
    if m:
        return (m.group(1), m.group(2), m.group(3))
    else:
        raise InvalidS3UriError(uri)

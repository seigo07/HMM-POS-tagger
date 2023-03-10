import sys

from pos_tagger import posTagger

INVALID_ARGS_NUMBER_ERROR = "Usage: python p1 <lang>"
INVALID_ARGS_LANG_ERROR = "Usage: en, fr, or uk for <lang>"

VALID_LANG_LIST = ["en", "fr", "uk"]
ARGV_NUMBER = 2


# Check <lang> is valid
def is_lang_valid():
    return sys.argv[1] in VALID_LANG_LIST


# Validate the number of args in arg.
if len(sys.argv) != ARGV_NUMBER:
    exit(INVALID_ARGS_NUMBER_ERROR)

# Validate language in arg.
if not is_lang_valid():
    exit(INVALID_ARGS_LANG_ERROR)

print("Loading...")
tagger = posTagger(sys.argv[1])
tagger.start()

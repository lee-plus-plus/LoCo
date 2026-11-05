import sys
# import pretty_errors


# Suppress stack trace for KeyboardInterrupt in pretty_errors
def handle_keyboard_interrupt(exc_type, exc_value, exc_traceback):
    if exc_type == KeyboardInterrupt:
        sys.stderr.write("\033[91m\nKeyboardInterrupt\033[0m\n")
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_keyboard_interrupt

# # Enable pretty_errors for better error traceback formatting
# pretty_errors.activate()

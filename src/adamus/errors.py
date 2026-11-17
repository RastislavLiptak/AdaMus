"""The one exception AdaMus raises for problems the operator can fix."""

LINE_END = chr(10)


class AdamusError(Exception):
    """Bad configuration, missing file, generator or headset that does not respond.

    Such a problem is shown to the operator as a message, without a traceback;
    every other exception is a fault of the program. Each argument is one line
    of the message.
    """

    def __str__(self) -> str:
        return LINE_END.join(str(line) for line in self.args)

class EDU(object):
    """ a class representing the EDU (id, span start and end, file) """

    def __init__(self, id, start, end, file):
        self.id = id
        self.start = start
        self.end = end
        self.file = file

    def __str__(self):
        return "EDU {}: ({}, {}) from {}".format(self.id, int(self.start), int(self.end), self.file)

    def __repr__(self):
        return "EDU {}: ({}, {}) from {}".format(self.id, int(self.start), int(self.end), self.file)

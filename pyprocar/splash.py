import pyprocar


def welcome():
    print(
        " ____        ____\n|  _ \ _   _|  _ \ _ __ ___   ___ __ _ _ __ \n| |_) | | | | |_) | '__/ _ \ / __/ _` | '__|\n|  __/| |_| |  __/| | | (_) | (_| (_| | |   \n|_|    \__, |_|   |_|  \___/ \___\__,_|_|\n       |___/"
    )
    print("A Python library for electronic structure pre/post-processing.\n")
    print("Version %s created on %s\n" % (pyprocar.__version__, pyprocar.__date__))
    print(
        "Please cite:\n \
Uthpala Herath, Pedram Tavadze, Xu He, Eric Bousquet, Sobhit Singh, Francisco Muñoz and Aldo Romero.,\n \
PyProcar: A Python library for electronic structure pre/post-processing.,\n \
Computer Physics Communications 251 (2020):107080.\n"
    )

    return

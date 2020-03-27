import pyfiglet
import pyprocar


def welcome():
    print(pyfiglet.figlet_format("PyProcar"))
    print("A Python library for electronic structure pre/post-processing.\n")
    print("Version %s created on %s\n" % (pyprocar.__version__, pyprocar.__date__))
    print(
        "Please cite: Herath, U., Tavadze, P., He, X., Bousquet, E., Singh, S., Muñoz, F. & Romero,\
    A., PyProcar: A Python library for electronic structure pre/post-processing.,\
    Computer Physics Communications 251 (2020):107080.\n"
    )

    return

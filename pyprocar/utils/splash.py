import logging

import pyprocar

user_logger = logging.getLogger("user")

def welcome():
    user_logger.info(
        " ____        ____\n|  _ \ _   _|  _ \ _ __ ___   ___ __ _ _ __ \n| |_) | | | | |_) | '__/ _ \ / __/ _` | '__|\n|  __/| |_| |  __/| | | (_) | (_| (_| | |   \n|_|    \__, |_|   |_|  \___/ \___\__,_|_|\n       |___/"
    )
    user_logger.info("A Python library for electronic structure pre/post-processing.\n")
    user_logger.info("Version %s created on %s\n" % (pyprocar.__version__, pyprocar.__date__))
    user_logger.info(
        "Please cite:\n\
- Uthpala Herath, Pedram Tavadze, Xu He, Eric Bousquet, Sobhit Singh, Francisco Muñoz and Aldo Romero.,\n \
 PyProcar: A Python library for electronic structure pre/post-processing.,\n \
 Computer Physics Communications 251, 107080 (2020).\n"
    )
    user_logger.info(
        "\
- L. Lang, P. Tavadze, A. Tellez, E. Bousquet, H. Xu, F. Muñoz, N. Vasquez, U. Herath, and A. H. Romero,\n \
 Expanding PyProcar for new features, maintainability, and reliability.,\n \
 Computer Physics Communications 297, 109063 (2024)."
    )

    dev_string = """
Developers:
- Francisco Muñoz
- Aldo Romero
- Sobhit Singh
- Uthpala Herath
- Pedram Tavadze
- Eric Bousquet
- Xu He
- Reese Boucher
- Logan Lang
- Freddy Farah
    """
    user_logger.info(dev_string)

    return

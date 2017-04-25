# -*- coding: utf-8 -*-
import platform
import os
import sys

def getHome():
    """
    return the home directory according to the platform
    :return:
    """
    system = platform.system()
    if system.startswith("Lin"):
        HOME = os.path.expanduser('~')
    elif system.startswith("Win"):
        HOME = r"C:\Users\SI30YD"
        if not os.path.exists(HOME):
            HOME = r"C:\Users\KH44IM"
    else:
        print "Unknown platform"
        sys.exit(0)
    return HOME
"""Copyright 2019 Philips Research. All rights reserved.

Author: Thomas Hagebols

Config file that contains paths to datasets, login credentials etc

This file should not be synced to git. In the template it is added to
.gitignore to prevent credentials from being stored on Git.
"""

from pathlib import Path

# Notice that we use forward slashes (even on Windows)
# We are using pathlib since it is a lot simpler than os.path
# See: https://realpython.com/python-pathlib/
DATA_PATHS = {'pivotal': Path('/home/2015-0100_pwsl_pivotal'),
              'dry_dry_sham': Path('/home/2015-0100_pwsl_pivotal/_FullFrontalStudy/EZDB')}


# Credentials
USERNAME = "some_username"
PASSWORD = "super_secret_password"

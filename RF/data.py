import json
# import logging
import math
import os
import pickle
import random
import string
import subprocess
from collections import defaultdict
from os import walk, path
from shutil import copy2

import numpy


# log = logging.getLogger('log')


def clean_domain_list(domain_list: list, dga=False):
    """
    Cleans a given domain list from invalid domains and cleans each single domain in the list.
    :param domain_list:
    :param dga:
    :return:
    """

    domain_list = [d.strip().lower() for d in domain_list]
    domain_list = list(filter(None, domain_list))

    if dga:
        # some ramnit domains ending with the pattern: [u'.bid', u'.eu']
        to_remove = []
        for d in domain_list:
            if '[' in d:
                # log.verbose('Domain contains [: {!s}'.format(d))
                to_remove.append(d)
                res = set()
                bracket_split = d.split('[')
                tlds = bracket_split[1].split(',')
                for tld in tlds:
                    tld = tld.strip()
                    tld = tld.split("'")[1].replace('.', '')
                    res_d = bracket_split[0] + tld
                    res.add(res_d)
                    # log.verbose('Cleaned domain: {!s}'.format(res_d))
                    domain_list.append(res_d)

        domain_list = [d for d in domain_list if d not in to_remove]

    return domain_list


def serialize_keep_copy(where, what, keep_copy=True):
    """
    Pickles given py obj. (what) to given file (where)
    If file exists: keeps a copy if not turned of via keep_copy=False
    :param where:
    :param what:
    :param keep_copy:
    :return:
    """
    if os.path.isfile(where):
        if not keep_copy:
            return
        where += '_copy_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    with open(where, 'wb') as f:
        pickle.dump(what, f)


class PublicSuffixes:
    """
    Represents the official public suffixes list maintained by Mozilla  https://publicsuffix.org/list/
    """
    def __init__(self, file='C:/Users/sxy/Desktop/experiment/RF/public_suffix.txt'):
        with open(file, encoding='utf-8') as f:
            self.data = f.readlines()

        self.data = clean_domain_list(self.data)
        self.data = ['.' + s for s in self.data if not (s.startswith('/') or s.startswith('*'))]
        self.data = clean_domain_list(self.data)

    def get_valid_tlds(self):
        return [s for s in self.data if len(s.split('.')) == 2]

    def get_valid_public_suffixes(self):
        return self.data

import pickle
import torch
import numpy as np


def consecutive(data, mode='first', stepsize=1):
    group = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    group = [item for item in group if len(item) > 0]

    if mode == 'first':
        result = [l[0] for l in group]
    elif mode == 'last':
        result = [l[-1] for l in group]
    return result


def word_segmentation(mat,
                      separator_idx={'th': [1, 2], 'en': [3, 4]},
                      separator_idx_list=[1, 2, 3, 4]):
    result = []
    sep_list = []
    start_idx = 0
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0:
            mode = 'first'
        else:
            mode = 'last'
        a = consecutive(np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [[item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]:
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]:
                if sep_lang == lang:
                    new_sep_pair = [lang, [sep_start_idx+1, sep[0]-1]]
                    if sep_start_idx > start_idx:
                        result.append(['', [start_idx, sep_start_idx-1]])
                    start_idx = sep[0]+1
                    result.append(new_sep_pair)
                else:
                    sep_lang = ''

    if start_idx <= len(mat)-1:
        result.append(['', [start_idx, len(mat)-1]])
    return result


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, separator_list={}, dict_pathlist={}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character
        self.separator_list = separator_list

        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep

        self.ignore_idx = [0] + [i+1 for i, item in enumerate(separator_char)]

        dict_list = {}
        for lang, dict_path in dict_pathlist.items():
            with open(dict_path, "rb") as input_file:
                word_count = pickle.load(input_file)
            dict_list[lang] = word_count
        self.dict_list = dict_list

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] not in self.ignore_idx and \
                        (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

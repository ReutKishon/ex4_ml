import math


def shannon_entropy(hebrew_freq, name):
    res = 0
    for c in name:
        res += hebrew_freq [ c ] * math.log2(hebrew_freq [ c ])
    return res * -1


def characteristics_huffman_code(probability, code):
    length_of_code = [ len(k) for k in code ]

    mean_length = sum([ a * b for a, b in zip(length_of_code, probability) ])

    print("Average length of the code: %f" % mean_length)


def position(probability, value, index):
    for j in range(len(probability)):
        if value >= probability [ j ]:
            return j
    return index - 1


def compute_code(probability: list [ float ]):
    num = len(probability)
    huffman_code = [ '' ] * num

    for i in range(num - 2):
        val = concat_huffman_bin_values(huffman_code, i, num, probability)

        pos = position(probability, val, i)
        probability = probability [ 0:(len(probability) - 2) ]
        probability.insert(pos, val)
        if isinstance(huffman_code [ num - i - 2 ], list) and isinstance(huffman_code [ num - i - 1 ], list):
            complete_code = huffman_code [ num - i - 1 ] + huffman_code [ num - i - 2 ]
        elif isinstance(huffman_code [ num - i - 2 ], list):
            complete_code = huffman_code [ num - i - 2 ] + [ huffman_code [ num - i - 1 ] ]
        elif isinstance(huffman_code [ num - i - 1 ], list):
            complete_code = huffman_code [ num - i - 1 ] + [ huffman_code [ num - i - 2 ] ]
        else:
            complete_code = [ huffman_code [ num - i - 2 ], huffman_code [ num - i - 1 ] ]

        huffman_code = huffman_code [ 0:(len(huffman_code) - 2) ]
        huffman_code.insert(pos, complete_code)

    huffman_code [ 0 ] = [ '0' + symbol for symbol in huffman_code [ 0 ] ]
    huffman_code [ 1 ] = [ '1' + symbol for symbol in huffman_code [ 1 ] ]

    if len(huffman_code [ 1 ]) == 0:
        huffman_code [ 1 ] = '1'

    count = 0
    final_code = [ '' ] * num

    for i in range(2):
        for j in range(len(huffman_code [ i ])):
            final_code [ count ] = huffman_code [ i ] [ j ]
            count += 1

    final_code = sorted(final_code, key=len)
    return final_code


def concat_huffman_bin_values(huffman_code, i, num, probability):
    val = probability [ num - i - 1 ] + probability [ num - i - 2 ]
    if huffman_code [ num - i - 1 ] != '' and huffman_code [ num - i - 2 ] != '':
        huffman_code [ -1 ] = [ '1' + symbol for symbol in huffman_code [ -1 ] ]
        huffman_code [ -2 ] = [ '0' + symbol for symbol in huffman_code [ -2 ] ]
    elif huffman_code [ num - i - 1 ] != '':
        huffman_code [ num - i - 2 ] = '0'
        huffman_code [ -1 ] = [ '1' + symbol for symbol in huffman_code [ -1 ] ]
    elif huffman_code [ num - i - 2 ] != '':
        huffman_code [ num - i - 1 ] = '1'
        huffman_code [ -2 ] = [ '0' + symbol for symbol in huffman_code [ -2 ] ]
    else:
        huffman_code [ num - i - 1 ] = '1'
        huffman_code [ num - i - 2 ] = '0'
    return val


def fill_probabilities(freq):
    freq = sorted(freq.items(), reverse=True, key=lambda x: x [ 1 ])
    probabilities: list [ float ] = sorted([ frequency [ 1 ] for frequency in freq ], reverse=True)
    return freq, probabilities


def print_encoded_first_name(results_dict, hebrew_letters_freq):
    while True:
        try:
            first_name1 = input("please enter your first name: ")
            s = ''.join(map(str, [ results_dict [ c ] for c in first_name1 ]))

        except KeyError:
            print(
                """Did you enter the name in hebrew or are there spaces of final letters in it?\n 
                    please try to run again""")
        else:
            break
    print(f"in huffman code:{s}")
    # huffman_on_name(first_name1)
    print(f"shannon entropy = {shannon_entropy(hebrew_letters_freq, first_name1)}")


def huffman_on_name(first_name1):
    print("using statistics based on name:")
    freq_dict = {}
    for c in first_name1:
        freq_dict [ c ] = freq_dict.get(c, 0) + 1
    freq, probabilities = fill_probabilities(freq_dict)
    huffman_code = compute_code(probabilities)
    results_dict_name = {}
    for idx, char in enumerate(freq):
        if huffman_code [ idx ] == '':
            continue
        results_dict_name [ char [ 0 ] ] = huffman_code [ idx ]
    for c in first_name1:
        print(results_dict_name [ c ], end="")
    print()
    print(shannon_entropy(freq_dict, first_name1))


def run():
    hebrew_letters_freq_dict = {'א': 0.0634, 'ב': 0.0474, 'ג': 0.013, 'ד': 0.0259, 'ה': 0.1087, 'ו': 0.1038,
                                'ז': 0.0133,
                                'ח': 0.0248,
                                'ט': 0.0124,
                                'י': 0.1106, 'כ': 0.027 + 0.0081, 'ל': 0.0739, 'מ': 0.0303 + 0.0459,
                                'נ': 0.011 + 0.0286,
                                'ס': 0.0148,
                                'ע': 0.0323,
                                'פ': 0.0027 + 0.0169, 'צ': 0.0012 + 0.0124, 'ק': 0.0214, 'ר': 0.0561, 'ש': 0.0441,
                                'ת': 0.0501}
    hebrew_letters_freq_list, probabilities = fill_probabilities(hebrew_letters_freq_dict)
    huffman_code = compute_code(probabilities)
    print('אות | Huffman code ')
    print('----------------------')
    results_dict = {}
    for idx, char in enumerate(hebrew_letters_freq_list):
        if huffman_code [ idx ] == '':
            print(" %-4r |%12s" % (char [ 0 ], 1))
            continue
        print(' %-4r |%12s' % (char [ 0 ], huffman_code [ idx ]))
        results_dict [ char [ 0 ] ] = huffman_code [ idx ]
    characteristics_huffman_code(probabilities, huffman_code)
    for _ in range(2):
        print_encoded_first_name(results_dict, hebrew_letters_freq_dict)
    exit()


if __name__ == "__main__":
    run()

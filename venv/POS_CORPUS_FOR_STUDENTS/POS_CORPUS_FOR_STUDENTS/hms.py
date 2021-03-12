'''
This is a bare bones hidden markov model for part of speech tagging written quickly in very bad style.
'''
import copy
from itertools import groupby

class HiddenMarkov:
    def __init__(self):
        self.emission_count = {}
        self.transition_count = {}
        self.transition_matrix = []
        self.pos_list = []

    def fill_counts(self, filename):
        with open(filename, 'r') as pos_trainer:
            text_lines = pos_trainer.readlines()

        pos_set = set()
        last_pos = None
        for line in text_lines:
            tokens = line.split()
            if not tokens:
                continue
            else:
                word, pos = tokens[0], tokens[1]
                pos_set.add(pos)
                if word in self.emission_count.keys():
                    self.emission_count[word].append(pos)
                else:
                    self.emission_count[word] = [pos]

                if pos not in self.transition_count.keys():
                    self.transition_count[pos] = [0]
                if last_pos is not None:
                    self.transition_count[pos].append(last_pos)

                '''
                Unlike with words in the emission count, the length of the transmission_count value associated with a
                pos key does not encode the number of appearances of the pos. 
                '''
                self.transition_count[pos][0] += 1
                last_pos = pos

        self.pos_list = list(pos_set)
        for x, row_word in enumerate(self.pos_list):
            new_row = []
            for y, col_word in enumerate(self.pos_list):
                new_row.append(self.transition_count[row_word].count(col_word) / self.transition_count[row_word][0])
            self.transition_matrix.append(new_row)


    def transmission(self, pos1, pos2):
        x = self.pos_list.index(pos1)
        y = self.pos_list.index(pos2)
        return self.transition_matrix[x][y]
        #return self.transition_count[pos1].count(pos2) / self.transition_count[pos1][0]

    def emission(self, word, pos):
        if word not in self.emission_count.keys():
            return 1/len(self.pos_list)
        return self.emission_count[word].count(pos) / len(self.emission_count[word])

    def viterbi(self, document):
        last_best = [(pos, [0, []]) for pos in self.pos_list]
        last_best = {pos: best for (pos, best) in last_best}

        for y1 in last_best.keys():
            last_best[y1][0] = self.emission(document[0], y1)
            last_best[y1][1] = [y1]
        sentence = document[1:]
        vocab_length = len(self.emission_count.keys())

        pos_list = last_best.keys()
        i = 0
        for word in sentence:
            new_best = copy.deepcopy(last_best)
            #print(i)
            i += 1
            for y1 in pos_list:
                emission_prob = self.emission(word, y1) + (1/vocab_length)
                best_prob = 0
                best_path = []
                for y0 in pos_list:
                    path_prob = (self.transmission(y1, y0) * emission_prob) + (1/vocab_length)
                    last_prob, last_path = last_best[y0]
                    path_prob *= last_prob
                    if path_prob >= best_prob:
                        best_prob = path_prob
                        best_path = last_path
                new_best[y1] = (best_prob * 2, best_path + [y1])

            last_best = new_best

        best_prob = 0
        best_path = []
        for last_label in last_best.keys():
            prob = last_best[last_label][0]
            if prob > best_prob:
                best_path = last_best[last_label][1]
                best_prob = prob

        return best_path

    def pos_tagger(self, filename):
        with open(filename, 'r') as doc:
            text_lines = doc.readlines()

        text_lines = [line.replace('\n', '') for line in text_lines]

        tags = []
        sentence = []
        for line in text_lines:
            if line == '':
                tags += self.viterbi(sentence)
                sentence = []
            else:
                sentence.append(line)

        if sentence:
            tags += self.viterbi(sentence)

        output_lines = []
        i = 0
        for line in text_lines:
            if not line:
                output_lines.append('')
            else:
                output_lines.append(line + '\t' + tags[i])
                i += 1

        with open('output.pos', 'w') as result:
            for line in output_lines:
                result.write(line + '\n')




example = HiddenMarkov()
#example.fill_counts('POS_train.pos')
#example.pos_tagger('POS_dev.words')

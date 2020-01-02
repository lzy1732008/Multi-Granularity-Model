
def feed_data_1(model, a_word,b_word,c_word,y_batch,dropout_rate):
    feed_dict = {
        model.input_X1: a_word,
        model.input_X2: b_word,
        model.x2_label: c_word,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }

    return feed_dict

def feed_data_2(model, a_word,b_word,c_word,y_batch,dropout_rate):
    feed_dict = {
        model.input_X1: a_word,
        model.input_X2: b_word,
        model.input_X3: c_word,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }

    return feed_dict

def feed_data_3(model, a_word,b_word,seq_a, seq_b,y_batch,dropout_rate):
    feed_dict = {
        model.input_X1: a_word,
        model.input_X2: b_word,
        model.seq1_len: seq_a,
        model.seq2_len: seq_b,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }
    return feed_dict

def feed_data_4(model, a_word,b_word,y_batch,dropout_rate):
    feed_dict = {
        model.input_X1: a_word,
        model.input_X2: b_word,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }
    return feed_dict

def feed_data_5(model, a_word,b_word,align_matrix, y_batch,dropout_rate):
    feed_dict = {
        model.input_X1: a_word,
        model.input_X2: b_word,
        model.align_matrix: align_matrix,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }
    return feed_dict


def feed_data_6(model, a_word,b_word,c_word,y_batch,dropout_rate):
    feed_dict = {
        model.input_X1: a_word,
        model.input_X2: b_word,
        model.ks: c_word,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }

    return feed_dict
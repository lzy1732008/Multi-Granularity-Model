
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

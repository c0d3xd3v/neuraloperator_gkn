from hlp.nn import load_check_point


if __name__== "__main__":

    dataset_path = 'data/train_data.h5'
    checkpoint_path = 'data/checkpoint.pt'

    model, _, _ = load_check_point(checkpoint_path)
    print(model)

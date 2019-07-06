import tensorflow as tf
import argparse
from data import mergeData
from model import multiTaskModel

def tuning_parameters(
        data_path,
        learning_rate=0.001,
        num_classes=4,
        max_epochs=10,
        display_step=100,
        batch_size=128,
        dropout_ratio=0.0,
        train_ratio=0.7,
        val_ratio=0.15,
        dense_units=128,
        lstm_units=64,
        lstm_num_layers=1,
        pos_weight=1.0
                      ):

    tf.reset_default_graph()

    with tf.Session() as sess:
        data = mergeData(
            path = data_path,
            num_classes = num_classes,
            batch_size = batch_size,
            train_ratio = train_ratio,
            val_ratio = val_ratio,
            display_step=100,
            useless_columns = ["stock_code", "time", "time_rank", "fin_rank", "news_rank", "time_rank_x", "time_rank_y"],
            target_news = "mood",
            target_fin = "st"
        )

        input_dimenion = data.input_dimension
        max_steps = data.max_steps
        multi_task_model = multiTaskModel(
            sess = sess,
            max_steps = max_steps,
            input_dimenion = input_dimenion,
            learning_rate=learning_rate,
            num_classes=num_classes,
            max_epochs=max_epochs,
            display_step=display_step,
            batch_size=batch_size,
            dropout_ratio=dropout_ratio,
            dense_units=dense_units,
            lstm_units = lstm_units,
            lstm_num_layers = lstm_num_layers,
            pos_weight = pos_weight
        )

        test_pred, test_y, test_loss, test_acc = multi_task_model.train(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", help="learning rate", required=False, default=0.001, type=float)
    parser.add_argument("--num_classes", help="number of classes", required=False, default=4, type=int)
    parser.add_argument("--max_epochs", help="max epochs", required=False, default=10, type=int)
    parser.add_argument("--display_step", help="number of steps to display", required=False, default=100, type=int)
    parser.add_argument("--batch_size", help="batch size", required=False, default=64, type=int)
    parser.add_argument("--dropout_ratio", help="dropout ratio", required=False, default=0.0, type=float)
    parser.add_argument("--train_ratio", help="training ratio", required=False, default=0.7, type=float)
    parser.add_argument("--val_ratio", help="validation ratio", required=False, default=0.15, type=float)
    parser.add_argument("--data_path", help="data path", required=False, default="./Data/merge_fin_news_try.pkl", type=str)
    parser.add_argument("--dense_units", help="dense_units", required=False, default=128, type=int)
    parser.add_argument("--lstm_units", help="number of lstm units", required=False, default=64, type=int)
    parser.add_argument("--lstm_num_layers", help="number of lstm layers", required=False, default=1, type=int)
    parser.add_argument("--pos_weight", help="positive weights", required=False, default=1.0, type=float)

    args = parser.parse_args()

    # train and test
    tuning_parameters(
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        max_epochs=args.max_epochs,
        display_step=args.display_step,
        batch_size=args.batch_size,
        dropout_ratio=args.dropout_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        dense_units=args.dense_units,
        lstm_units=args.lstm_units,
        lstm_num_layers=args.lstm_num_layers,
        pos_weight=args.pos_weight
    )

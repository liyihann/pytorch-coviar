"""Combine testing results of the three models to get final accuracy."""

import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="combine predictions")
    parser.add_argument('--iframe', type=str, required=True,
                        help='iframe score file.')
    parser.add_argument('--mv', type=str, required=True,
                        help='motion vector score file.')
    # -----------------------------ORIGINAL_CODE_START-----------------------------
    # parser.add_argument('--res', type=str, required=True,
    #                     help='residual score file.')
    # -----------------------------ORIGINAL_CODE_END-------------------------------
    # -----------------------------MODIFIED_CODE_START-----------------------------
    parser.add_argument('--res', type=str, required=False,
                        help='residual score file.')
    # -----------------------------MODIFIED_CODE_END-------------------------------


    parser.add_argument('--wi', type=float, default=2.0,
                        help='iframe weight.')
    parser.add_argument('--wm', type=float, default=1.0,
                        help='motion vector weight.')
    parser.add_argument('--wr', type=float, default=1.0,
                        help='residual weight.')

    args = parser.parse_args()

    # -----------------------------ORIGINAL_CODE_START-----------------------------
    # with np.load(args.iframe) as iframe:
    #     with np.load(args.mv) as mv:
    #         with np.load(args.res) as residual:
    #             n = len(mv['names'])
    #
    #             i_score = np.array([score[0][0] for score in iframe['scores']])
    #             mv_score = np.array([score[0][0] for score in mv['scores']])
    #             res_score = np.array([score[0][0] for score in residual['scores']])
    #
    #             i_label = np.array([score[1] for score in iframe['scores']])
    #             mv_label = np.array([score[1] for score in mv['scores']])
    #             res_label = np.array([score[1] for score in residual['scores']])
    #             assert np.alltrue(i_label == mv_label) and np.alltrue(i_label == res_label)
    #
    #             combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
    #
    #             accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
    #             print('Accuracy: %f (%d).' % (accuracy, n))
    # -----------------------------ORIGINAL_CODE_END-------------------------------
    # -----------------------------MODIFIED_CODE_START-----------------------------
    if args.res:
        with np.load(args.iframe, allow_pickle=True) as iframe:
            with np.load(args.mv, allow_pickle=True) as mv:
                with np.load(args.res, allow_pickle=True) as residual:
                    n = len(mv['names'])

                    i_score = np.array([score[0][0] for score in iframe['scores']])
                    mv_score = np.array([score[0][0] for score in mv['scores']])
                    res_score = np.array([score[0][0] for score in residual['scores']])

                    i_label = np.array([score[1] for score in iframe['scores']])
                    mv_label = np.array([score[1] for score in mv['scores']])
                    res_label = np.array([score[1] for score in residual['scores']])
                    assert np.alltrue(i_label == mv_label) and np.alltrue(i_label == res_label)

                    combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr

                    accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
                    print('Accuracy: %f (%d).' % (accuracy, n))
    else:
        with np.load(args.iframe, allow_pickle=True) as iframe:
            with np.load(args.mv, allow_pickle=True) as mv:
                    n = len(mv['names'])
                    i_score = np.array([score[0][0] for score in iframe['scores']])
                    mv_score = np.array([score[0][0] for score in mv['scores']])

                    i_label = np.array([score[1] for score in iframe['scores']])
                    mv_label = np.array([score[1] for score in mv['scores']])
                    assert np.alltrue(i_label == mv_label)
                    combined_score = i_score * args.wi + mv_score * args.wm
                    accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
                    print('Accuracy: %f (%d).' % (accuracy, n))
    # -----------------------------MODIFIED_CODE_END-------------------------------

if __name__ == '__main__':
    main()

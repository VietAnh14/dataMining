import model
import argparse


class Train(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        result = dict()
        try:
            val = float(values)
        except ValueError:
            print('Wrong argument exception, value error')
            return
        if values is None:
            print('Training ....')
            result = model.train()
        elif 0.1 <= float(values) < 1:
            print('Training ....')
            result = model.train(float(values))
        else:
            print('Wrong train size, train size should be > 0.5 and < 1')
            return
        print('Accuracy:')
        for item in result:
            print(item, result[item])
        setattr(namespace, self.dest, values)


# class Predict(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string):
#         sample = list()
#         try:
#             arr = [float(x) for x in values.split(',')]
#             sample.append(arr)
#         except ValueError:
#             print('Wrong sample, please read usage for more details')
#             return
#         if args.alg:
#             if args.alg == 'rt':
#                 predict = model.predict_grade(sample, 'random')
#                 print('Predict grade: {}'.format(predict[0].round(2)))
#             elif args.alg == 'lr':
#                 predict = model.predict_grade(sample, 'linear')
#                 print('Predict grade: {}'.format(predict[0].round(2)))
#
#             else:
#                 print('Wrong alg, please read usage for more information')
#         setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser(description='Data mining program project to predict student\'s grades')

parser.add_argument('-t', '--train', action=Train, metavar='', nargs="?", help='Train model, can specific train size,'
                                                                               'default size is 0.7')

parser.add_argument('-p', '--predict', action='store', metavar='',
                    help='-p <sample> Predict student\'s grade')

parser.add_argument('-alg', '--algorithm', dest='alg', default='lr', choices=['lr', 'rf'],
                    metavar='', help='Choose the algorithm to predict student\'s grade \'lr\' is linear regression, '
                                     '\'rf\' is random forest ex: python program.py -p 12,13,0,0,0,4 -alg rf')
args = parser.parse_args()


def predict_value(arr):
    sample = list()
    try:
        arr = [float(x) for x in arr.split(',')]
        sample.append(arr)
    except ValueError:
        print('Wrong sample, please read usage for more details')
        return
    if args.alg:
        if args.alg == 'rf':
            predict = model.predict_grade(sample, 'random')
            print('Predict grade: {}'.format(predict[0].round(2)))
        elif args.alg == 'lr':
            predict = model.predict_grade(sample, 'linear')
            print('Predict grade: {}'.format(predict[0].round(2)))

        else:
            print('Wrong alg {}, please read usage for more information'.format(args.alg))


if args.predict:
    predict_value(args.predict)

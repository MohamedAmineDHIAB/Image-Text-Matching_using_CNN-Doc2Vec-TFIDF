import numpy as np
import pandas as pd

"""
Parsing the arguments:
* path to directory with predictions and truths
* file name of file with predictions
* file name of file with truths

Example usage: 
python3 DataEvaluation.py -d "path" -p "predictions" -t "truths"
"""
def parse_arguments():
    global path
    global predictions
    global truths

    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-d', '--path', help='path to data directory',
                        required=True)
    parser.add_argument('-p', '--predictions', 
                        help='file name with predictions',
                        required=True)
    parser.add_argument('-t', '--truths', 
                        help='file name with truths',
                        required=True)

    args = parser.parse_args()
    path = args.path
    fileName_truths = args.truths
    fileName_predictions = args.predictions

def load_results(fileName, ext = 'csv'):
    results = []
    if (ext == 'csv'):
        res = pd.read_csv(fileName)
        ranked_list = res.columns[1]
        results = [list(map(int, res[ranked_list][i].translate({ord('['): None,ord(']'): None}).split(","))) for i in range(len(res))]
    else:
        f = open(fileName, "r")
        for line in f:
            results.append([int(item) for item in line.strip().split()])
        f.close()
    return results

def precisionAtN(pred, truth, N):
    precision = int(truth in pred[:N])
    return precision/N

def precisionAvg(results, truths, N = 3):
    if (N > np.shape(results)[1]):
        print("N is larger than number of positions in ranking.")
        return
    precision_sum = 0
    for i in range(len(truths)):
        precision_sum += precisionAtN(results[i], truths[i], N)  
    return precision_sum/len(truths)

def mrrAtN(pred, truth, N):
    mrr = 0
    if truth in pred[:N]:
        mrr = 1/(pred.index(truth)+1)   
    return mrr

def mrrAvg(results, truths, N = 3):
    if (N > np.shape(results)[1]):
        print("N is larger than number of positions in ranking.")
        return
    mrr_sum = 0
    for i in range(len(truths)):
        mrr_sum += mrrAtN(results[i], truths[i], N)
    return mrr_sum/len(truths)

def accuracy(results, truths):
    accuracy = 0
    for i in range(len(truths)):
        if truths[i] == results[i][0]:
            accuracy+=1
    return accuracy/len(truths)

def accuracyAtN(results, truths,N):
    accuracy = 0
    for i in range(len(truths)):
        if truths[i] in results[i][:N]:
            accuracy+=1
    return accuracy/len(truths)

if __name__ == "__main__":

	results_cosine = load_results(path.join(path, fileName), 'csv')
	truths = pd.read_csv(path.join(path, fileName))['iid']
	for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
		print(precisionAtN(results_cosine, truths, N))
		print(mrrAtN(results_cosine, truths, N))
		print(accuracyAtN(results_cosine, truths, N))


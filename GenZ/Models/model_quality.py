from typing import Optional
from typing import List
import csv


class QualityMetric():
    def __init__(self):
        pass
    def __repr__(self):
        variables = vars(self)
        return ', '.join(f'{name}: {value}' for name, value in variables.items())

    # def to_dict(self):
        # return vars(self)

class MMLU(QualityMetric):
    '''
    5-Shot MMLU Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'MMLU'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class GSM8K(QualityMetric):
    '''
    GSM8K Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'GSM8K'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class IFEval(QualityMetric):
    '''
    IFEval Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'IFEval'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class TLDR(QualityMetric):
    '''
    TLDR Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'TLDR'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class BIG_Bench(QualityMetric):
    '''
    BIG_Bench Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'BIG_Bench'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class GPQA(QualityMetric):
    '''
    GPQA Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'GPQA'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class Hellaswag(QualityMetric):
    '''
    Hellaswag Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'Hellaswag'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class TriviaQA(QualityMetric):
    '''
    TriviaQA Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'TriviaQA'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class MATH(QualityMetric):
    '''
    0-Shot MATH Accuracy
    '''
    def __init__(self, accuracy:float, shots:int=None):
        self.name = 'MATH'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class QualityMetricsCollection:
    def __init__(self, metrics: Optional[List[QualityMetric]] = None):
        self.metrics = metrics if metrics is not None else []

    def add_metric(self, metric: QualityMetric):
        self.metrics.append(metric)

    def to_dict(self):
        return {i: metric.__dict__ for i, metric in enumerate(self.metrics)}

    def __repr__(self):
        return '\n'.join(repr(metric) for metric in self.metrics)

def get_model_quality_collection(file, model_name):
    collection = QualityMetricsCollection()
    with open(file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            benchmark_name = row['Benchmark']
            accuracy = float(row[model_name])
            shots = int(row['Shots']) if 'Shots' in row and row['Shots'] else None
            if benchmark_name == 'MMLU':
                collection.add_metric(MMLU(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'GSM8K':
                collection.add_metric(GSM8K(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'IFEval':
                collection.add_metric(IFEval(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'TLDR':
                collection.add_metric(TLDR(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'BIG_Bench':
                collection.add_metric(BIG_Bench(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'GPQA':
                collection.add_metric(GPQA(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'Hellaswag':
                collection.add_metric(Hellaswag(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'TriviaQA':
                collection.add_metric(TriviaQA(accuracy=accuracy, shots=shots))
            elif benchmark_name == 'MATH':
                collection.add_metric(MATH(accuracy=accuracy, shots=shots))
    return collection
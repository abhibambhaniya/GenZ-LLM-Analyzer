from typing import Optional
from typing import List


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
    def __init__(self, accuracy:float, shots:int):
        self.name = 'MMLU'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class GSM8K(QualityMetric):
    '''
    GSM8K Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'GSM8K'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class IFEval(QualityMetric):
    '''
    IFEval Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'IFEval'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class TLDR(QualityMetric):
    '''
    TLDR Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'TLDR'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class BIG_Bench(QualityMetric):
    '''
    BIG_Bench Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'BIG_Bench'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class GPQA(QualityMetric):
    '''
    GPQA Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'GPQA'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class Hellaswag(QualityMetric):
    '''
    Hellaswag Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'Hellaswag'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class TriviaQA(QualityMetric):
    '''
    TriviaQA Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
        self.name = 'TriviaQA'
        self.accuracy = accuracy
        self.shots = shots
        self.metric_type = 'Accuracy'
        super().__init__()

class MATH(QualityMetric):
    '''
    0-Shot MATH Accuracy
    '''
    def __init__(self, accuracy:float, shots:int):
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

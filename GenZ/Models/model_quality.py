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
    def __init__(self, accuracy:float):
        self.name = 'MMLU'
        self.accuracy = accuracy
        self.metric_type = 'Accuracy'
        super().__init__()

class MATH(QualityMetric):
    '''
    0-Shot MATH Accuracy
    '''
    def __init__(self, accuracy:float):
        self.name = 'MATH'
        self.accuracy = accuracy
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
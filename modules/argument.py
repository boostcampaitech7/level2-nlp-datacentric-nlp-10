from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataCentric:
    train_route: Optional[str] = field(
        default = '/data/ephemeral/home/code/data/train.csv',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
    test_route: Optional[str] = field(
        default = '/data/ephemeral/home/code/data/test.csv',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
    submission_route: Optional[str] = field(
        default = '/data/ephemeral/home/code/data/sample_submission.csv',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
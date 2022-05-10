class CoverageResult:
    def __init__(self, cov_obj):
        self.covered = cov_obj.covered()
        self.total = cov_obj.total()
        self.percentage = cov_obj.percentage()


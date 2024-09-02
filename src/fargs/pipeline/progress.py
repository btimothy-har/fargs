from tqdm.asyncio import tqdm


class ProgressReporter:
    def __init__(self, name: str):
        self.pbar = None
        self.name = name

    def start(self, position: int = 0):
        self.pbar = tqdm(desc=self.name, leave=False, position=position)

    def next_step(self, step: str, total: int):
        curr_total = self.pbar.n
        self.pbar.reset(total=curr_total + total)
        self.pbar.update(curr_total)
        self.pbar.refresh()
        self.pbar.desc = f"{self.name[:20]} - {step}"

    def update(self, increment: int = 1):
        self.pbar.update(increment)

    def complete(self):
        self.pbar.desc = f"{self.name[:20]} - Completed"
        self.pbar.close()

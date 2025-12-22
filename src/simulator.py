import numpy as np
from typing import List, Literal, get_args, Tuple
from tqdm import tqdm
from dataclasses import dataclass


def binary_insert(a: np.array, b: np.array, side: Literal["left", "right"] = "left") -> np.array:
    """Binary insert of array b into array a

    Args:
        a (np.array): into
        b (np.array): what
        side (Literal['left', 'right'], optional): binary search insert side. Defaults to 'left'.
                                left:  a[i-1] <  v <= a[i]
                                right: a[i-1] <= v <  a[i]

    Returns:
        np.array: total array
    """
    return np.insert(a, a.searchsorted(b, side=side), b)


@dataclass
class BurstGI:
    """Parameters of model (Burst|GI|n, m)

    Attributes:
            K (int): number of repetitions
            n (int): query length
            m (int): services number
            burst_prob (float): burst probability
            service_policy (Literal['SJF']): service policy. Defaults to 'SJF'
            overflow_policy (Literal['Kill']): policy in case of queue overflow. Defaults to 'Kill'
            Burst_distr (Literal['Geom']): probability before birst distribution. Defaults to 'Geom'
            Burst_number_distr (Literal['Poiss']): number of applications in burst distribution. Defaults to 'Poiss'
            burst_mean (float): Burst_distr distribution mean. Defaults to 0
            burst_scale (float): Burst_distr distribution sclae. Defaults to 1
            A_distr (Literal['exp']): receipt applications time distribution. Defaults to 'exp'
            B_distr (Literal['exp']): service time distribution. Defaults to 'exp'
            a_mean (float): A_distr distribution mean. Defaults to 0
            b_mean (float): B_distr distribution mean. Defaults to 0
            a_scale (float): A_distr distribution scale. Defauts to 1
            b_scale (float): A_distr distribution scale. Defauts to 1


    > Distributions:
            * Exp -- exponential. Allowed parameters: scale
            * Poiss -- Poisson. Allowed parameters: scale
            * Geom -- geometric. Allowed parameters: probability

    > service policies:
            * SJF -- Shortest Job First
    """

    K: int
    n: int
    m: int
    burst_prob: float
    service_policy: Literal["SJF"] = "SJF"
    overflow_policy: Literal["Kill"] = "Kill"
    Burst_distr: Literal["Geom"] = "Geom"
    Burst_number_distr: Literal["Poiss"] = "Poiss"
    burst_mean: float = 0
    burst_scale: float = 1
    A_distr: Literal["Exp"] = "Exp"
    B_distr: Literal["Exp"] = "Exp"
    a_mean: float = 0
    b_mean: float = 0
    a_scale: float = 1
    b_scale: float = 1

    def __post_init__(self) -> None:
        """Logic validation and inition log

        Raises:
                ValueError: digit mismatch
                ValueError: allowance to some values list
        """
        # parameters logic validation
        if self.n < 0:
            raise ValueError("n must be >= 0!")
        if self.m <= 0:
            raise ValueError("m must be > 0!")
        if not (1 >= self.burst_prob >= 0):
            raise ValueError("Burst probability must be in [0, 1]")

        self.__validate_literals()

        # TODO: self.rng = random.Random(seed=42)
        # TODO: show all parameters when suit the system
        print(
            f"System initialized:\tBurst({self.Burst_distr}, {'M' if self.A_distr == 'Exp' else self.A_distr})/"
            f"{'M' if self.B_distr == 'Exp' else self.B_distr}/{self.n}, {self.m}\n"
        )

    def __validate_literals(self) -> None:
        """Automaticaly checks fields to be appropriate in specific list"""
        for field_name, field_type in self.__annotations__.items():
            # check if origin is Literal
            if getattr(field_type, "__origin__", None) is Literal:
                current_value = getattr(self, field_name)  # get current field
                allowed_values = get_args(field_type)  # list of allowed values
                if current_value not in allowed_values:
                    raise ValueError(
                        f"Field '{field_name}' has isn't appropriate type '{current_value}'! "
                        f"Allowed: {allowed_values}"
                    )

    def __initialize_system(self):
        # visit states accordingly to positions. Queue + services number + empty busy
        self.v = np.zeros(self.m + self.n + 1, int)
        # times in states accordingly to position. Queue + services number + empty busy
        self.times = np.zeros(self.m + self.n + 1, float)

        match self.A_distr:  # receipt applications time distribution
            case "Exp":
                self.arrival_times = np.random.exponential(self.a_scale, self.K)

        match self.B_distr:  # service time distribution
            case "Exp":
                service = np.random.exponential
                self.demands = list(np.random.exponential(self.b_scale, self.K).reshape((self.K, 1)))

        match self.Burst_distr:  # probability before birst distribution
            case "Geom":
                num_func = np.random.geometric

        match self.Burst_number_distr:  # number of applications in burst distribution
            case "Poiss":
                i = num_func(self.burst_prob)
                while i < self.K:
                    # if there 0 element from Poisson replace it with 1
                    self.demands[i] = service(self.b_scale,
                                              1 if (v := np.random.poisson(self.burst_scale)) == 0 else v,)
                    i += num_func(self.burst_prob)

    def __reset_state(self):
        self.workload_process: List[np.ndarray] = []
        self.wasted_demands: int = 0
        self.__initialize_system()

    def __renew_workload(self, wld: List[np.array], demand: np.array) -> List[np.array]:
        """Wrapper for default renewing workload operation

        Args:
            wld (List[np.array]): workload
            demand (np.array): demand(-s)

        Returns:
            List[np.array]: merged workload
        """
        wld = binary_insert(wld[self.m:], wld[: self.m])
        wld = binary_insert(wld, demand)
        if len(wld) > self.n + self.m:
            self.wasted_demands += len(wld) - (self.n + self.m)
        return wld[: self.n + self.m]  # TODO: here overflow_policy must be used

    def run_sjf(self) -> None:
        """Runs simulation with SJF policy"""
        workload: np.ndarray = None  # then this will be similar that workload has infinity in it
        for arrival_time, demand in tqdm(zip(self.arrival_times, self.demands)):
            # for correct binary inserting
            demand = np.sort(demand)

            # workload writing
            self.workload_process.append(workload)
            if workload is None:  # first iteration
                workload = demand[: self.n + self.m]

                # statistics collection
                self.times[0] += arrival_time
                self.v[0] += 1
                continue
            if arrival_time < workload[0]:
                workload[: self.m] -= arrival_time

                # statistics collection
                self.times[len(workload)] += arrival_time
                self.v[len(workload)] += 1

                # renewing
                workload = self.__renew_workload(workload, demand)
            elif arrival_time == workload[0]:
                workload[: self.m] -= workload[0]

                # statistics collection
                self.times[len(workload)] += arrival_time
                self.v[len(workload)] += 1

                # state transformation
                workload = workload[workload > 0]
                workload = self.__renew_workload(workload, demand)
            else:
                while len(workload) > 0 and arrival_time > workload[0]:
                    arrival_time -= workload[0]

                    # statistics collection
                    self.times[len(workload)] += workload[0]
                    self.v[len(workload)] += 1

                    # state transformation
                    workload[: self.m] -= workload[0]
                    workload = workload[workload > 0]

                    # workload writing
                    self.workload_process.append(workload)
                if len(workload) > 0 and arrival_time < workload[0]:
                    workload[: self.m] -= arrival_time

                    # statistics collection
                    self.times[len(workload)] += arrival_time
                    self.v[len(workload)] += 1

                    workload = self.__renew_workload(workload, demand)
                if len(workload) == 0 and arrival_time > 0:
                    # statistics collection
                    self.times[0] += arrival_time
                    self.v[len(workload)] += 1

                    # state transformation
                    workload = demand[: self.n + self.m]

        while len(workload) > 0:
            self.workload_process.append(workload)

            # statistics collection
            self.times[len(workload)] += workload[0]
            self.v[len(workload)] += 1

            # state transformation
            workload[: self.m] -= workload[0]
            workload = workload[workload > 0]

    def run(self, show_statistics: bool = False) -> Tuple[np.ndarray, np.ndarray, int, List[np.ndarray]]:
        """Runs simulation depends on input service policy

        Attributes:
            show_statistics (bool): show statistics in the end of simulation (True) or not (False). Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, List[np.ndarray]]: state times, state occurences,
                                                                  wasted demands, workload
        """
        self.__reset_state()
        match self.service_policy:
            case "SJF":
                self.run_sjf()

        if show_statistics:
            self.print_statistics()

        return self.times, self.v, self.wasted_demands, self.workload_process

    def print_statistics(self):
        """Prints statistics"""
        print()
        print(f"Wasted jobs:\t\t{self.wasted_demands}")
        print(f"States Frequencies:\t({', '.join(self.v.astype(str))})")
        print(f"Times distribution:\t({', '.join(self.times.round(2).astype(str))})")


if __name__ == "__main__":
    burst_model = BurstGI(5, 3, 2, 0.5)
    times, v, wasted_jobs, *_ = burst_model.run(show_statistics=True)

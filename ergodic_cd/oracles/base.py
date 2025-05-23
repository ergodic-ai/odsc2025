import pandas
import numpy
from typing import Callable, Literal, Optional, Union
from scipy.stats import f
from sklearn.preprocessing import PolynomialFeatures


def linear(
    X: Union[pandas.DataFrame, None],
    y: pandas.Series,
    node: Optional[str] = None,
    parent_set: Optional[set] = None,
):
    """Return the residual sum of squares for the data X and y."""

    if X is None:
        X_values = numpy.ones(shape=(y.shape[0], 1))

    else:
        X_values = X.values
        X_values = numpy.hstack([numpy.ones(shape=(X.shape[0], 1)), X_values])

    y_values: numpy.ndarray = y.values  # type: ignore

    _, [rss], _, _ = numpy.linalg.lstsq(X_values, y_values, rcond=None)

    p = X_values.shape[1]

    return rss, p


def cubic(
    X: Union[pandas.DataFrame, None],
    y: pandas.Series,
    node: Optional[str] = None,
    parent_set: Optional[set] = None,
):
    """Return the residual sum of squares for the data X and y."""

    if X is None:
        X_values = numpy.ones(shape=(y.shape[0], 1))

    else:
        X_values = X.values
        X_values = PolynomialFeatures(degree=3).fit_transform(X_values)

    y_values: numpy.ndarray = y.values  # type: ignore

    _, [rss], _, _ = numpy.linalg.lstsq(X_values, y_values, rcond=None)

    p = X_values.shape[1]

    return rss, p


def get_f_and_p_val(
    rss_full: float, rss_reduced: float, p_full: int, p_reduced: int, n: int
) -> tuple[float, float]:
    F_stat = ((rss_reduced - rss_full) / (p_full - p_reduced)) / (
        rss_full / (n - p_full)
    )
    p_value: float = 1 - f.cdf(F_stat, p_full - p_reduced, n - p_full)  # type: ignore

    return F_stat, p_value


def hashFeatureList(feature_list, target):
    return str(sorted(feature_list)) + target


class BaseOracle:
    def __init__(
        self,
        data: pandas.DataFrame,
        threshold: float = 0.05,
        operation: Literal["mm", "max", "min"] = "mm",
        rank: bool = False,
        learner: Callable = linear,
    ):
        self.data = data
        self.threshold = threshold
        self.RSSCache = {}
        self.LLCache = {}
        self.operation = operation
        self.learner = learner
        self.rank = rank

        if self.rank:
            self.data = self.data.rank(axis=0, method="average")

    def f_test(self, x: str, y: str, Z: Optional[list[str]] = []):
        """Perform an F-test to compare the full model with the reduced model."""

        x_features = [x]
        Z_features = Z or []
        y_feat = y

        y = self.data.loc[:, y].astype("float")
        n = self.data.shape[0]

        thisHash = hashFeatureList(x_features + Z_features, y_feat)
        if thisHash in self.RSSCache:
            RSS_full, p_full = self.RSSCache[thisHash]

        else:
            X = self.data[Z_features + x_features].copy()
            RSS_full, p_full = self.learner(X, y, y_feat, set(Z_features + x_features))
            self.RSSCache[thisHash] = RSS_full, p_full

        # Reduced model (without the last predictor)
        thisHash = hashFeatureList(Z_features, y_feat)
        if thisHash in self.RSSCache:
            RSS_reduced, p_reduced = self.RSSCache[thisHash]
        else:
            if len(Z_features) == 0:
                X_reduced = None
            else:
                X_reduced = self.data[Z_features].copy()

            RSS_reduced, p_reduced = self.learner(X_reduced, y, y_feat, set(Z_features))

            self.RSSCache[thisHash] = RSS_reduced, p_reduced

        return get_f_and_p_val(RSS_full, RSS_reduced, p_full, p_reduced, n)

    def _run_both_ways(self, x: str, y: str, Z: Optional[list[str]] = []):
        F1, p1 = self.f_test(x, y, Z)
        F2, p2 = self.f_test(y, x, Z)

        if self.operation == "mm":
            return min(2 * min(p1, p2), max(p1, p2))
        elif self.operation == "max":
            return max(p1, p2)
        elif self.operation == "min":
            return min(p1, p2)

        else:
            raise ValueError("Invalid operation: ", self.operation)

    def _run(self, x: str, y: str, Z: Optional[list[str]] = []):
        assert y in self.data.columns, f"Target {y} not in data columns"
        assert x in self.data.columns, f"Source {x} not in data columns"
        for z in Z or []:
            assert (
                z in self.data.columns
            ), f"Conditioning variable {z} not in data columns"

        return self._run_both_ways(x, y, Z)

    def __call__(self, x: str, y: str, Z: Optional[list[str]] = []):
        # print(f"Running test for {x} -> {y} | {Z}")
        # if Z is not a list (a tuple or a set), convert it to a list
        if Z is None:
            Z = []
        if isinstance(Z, tuple) or isinstance(Z, set):
            Z = list(Z)

        return bool(self._run(x, y, Z) > self.threshold)

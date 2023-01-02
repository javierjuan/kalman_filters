import json
import os
from typing import Union, Sequence, Dict, Any, Optional

import numpy as np


class KalmanFilter:
    def __init__(self, dx: int, dz: int, dt: float = 1, measurement_std: float = 1, process_std: float = 1, **kwargs):
        self.measurement_std = measurement_std
        self.process_std = process_std

        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.df = int(dx / dz)

        self.x = np.zeros([self.dx, 1])
        self.y = np.zeros([self.dx, 1])
        self.B = None
        self.F = np.zeros([self.dx, self.dx])
        self.G = np.zeros([self.dz, self.dx])
        self.H = np.zeros([self.dz, self.dx])
        self.P = np.eye(self.dx)
        self.Q = np.zeros([self.dx, self.dx])
        self.R = np.eye(self.dz) * measurement_std ** 2
        self.S = np.zeros([self.dz, self.dz])
        self.K = np.zeros([self.dx, self.dz])

        self._Ix = np.eye(self.dx)
        self._Iz = np.eye(self.dz)
        self.initialized = False

    def to_json(self, file_path=None, indent=4) -> str:
        output = json.dumps(self.to_dict(), indent=indent)
        if file_path is not None:
            with open(file_path, 'w') as file:
                file.write(output)
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {'dx': self.dx, 'dz': self.dz, 'dt': self.dt, 'measurement_std': self.measurement_std,
                'process_std': self.process_std}

    @classmethod
    def from_json(cls, inputs):
        if isinstance(inputs, str):
            if os.path.isfile(inputs):
                with open(inputs, 'r') as file:
                    inputs = json.load(file)
            else:
                inputs = json.loads(inputs)
        return cls(**inputs)

    @staticmethod
    def factory(motion: str, dimensions: int, **settings):
        if motion == 'constant_velocity':
            if dimensions == 1:
                return KalmanFilterConstantVelocity1D(**settings)
            elif dimensions == 2:
                return KalmanFilterConstantVelocity2D(**settings)
            elif dimensions == 3:
                return KalmanFilterConstantVelocity3D(**settings)
            elif dimensions == 4:
                return KalmanFilterConstantVelocity4D(**settings)
            else:
                return KalmanFilterConstantVelocity(**settings)
        elif motion == 'constant_acceleration':
            if dimensions == 1:
                return KalmanFilterConstantAcceleration1D(**settings)
            elif dimensions == 2:
                return KalmanFilterConstantAcceleration2D(**settings)
            elif dimensions == 3:
                return KalmanFilterConstantAcceleration3D(**settings)
            elif dimensions == 4:
                return KalmanFilterConstantAcceleration4D(**settings)
            else:
                return KalmanFilterConstantAcceleration(**settings)
        else:
            return KalmanFilter(**settings)

    def initialize(self, z: Sequence, std: float):
        self.x = self.measurement_to_state(z)
        _P = np.kron(self._Iz, 1 / np.arange(self.df, 0, -1))
        self.P = np.linalg.multi_dot([_P.T, self._Iz, _P]) * std ** 2
        self.initialized = True

    def predict(self, u: Optional[Union[int, float, Sequence]] = None) -> np.array:
        if not self.initialized:
            raise ValueError('KalmanFilter not initialized. Please call `initialize(x: Sequence, std: float)` before '
                             'calling `predict()` or `update()` methods')
        self.x = np.dot(self.F, self.x)
        if self.B is not None and u is not None:
            self.x += np.dot(self.B, u)
        self.P = np.linalg.multi_dot([self.F, self.P, self.F.T]) + self.Q
        return self.x

    def predict_steady_state(self, u: Optional[Union[int, float, Sequence]] = None) -> np.array:
        if not self.initialized:
            raise ValueError('KalmanFilter not initialized. Please call `initialize(x: Sequence, std: float)` before '
                             'calling `predict()` or `update()` methods')
        self.x = np.dot(self.F, self.x)
        if self.B is not None and u is not None:
            self.x += np.dot(self.B, u)
        return self.x

    def update(self, z: Sequence) -> np.array:
        if not self.initialized:
            raise ValueError('KalmanFilter not initialized. Please call `initialize(x: Sequence, std: float)` before '
                             'calling `predict()` or `update()` methods')

        z = np.asarray(z)
        z = np.expand_dims(z, axis=1) if z.ndim == 1 else z

        PHT = np.dot(self.P, self.H.T)
        self.S = np.dot(self.H, PHT) + self.R
        self.K = np.dot(PHT, np.linalg.inv(self.S))

        self.y = z - np.dot(self.H, self.x)
        self.x += np.dot(self.K, self.y)
        IKH = self._Ix - np.dot(self.K, self.H)
        self.P = np.linalg.multi_dot([IKH, self.P, IKH.T]) + np.linalg.multi_dot([self.K, self.R, self.K.T])
        return self.x

    def update_steady_state(self, z: Sequence) -> np.array:
        if not self.initialized:
            raise ValueError('KalmanFilter not initialized. Please call `initialize(x: Sequence, std: float)` before '
                             'calling `predict()` or `update()` methods')

        z = np.asarray(z)
        z = np.expand_dims(z, axis=1) if z.ndim == 1 else z

        self.y = z - np.dot(self.H, self.x)
        self.x += np.dot(self.K, self.y)
        return self.x

    def state_to_measurement(self, x: Sequence) -> np.array:
        return np.dot(self.H, x)

    def measurement_to_state(self, z: Sequence) -> np.array:
        state = np.kron(z, self.H[0, :self.df])
        return np.expand_dims(state, axis=1) if state.ndim == 1 else state

    def log_likelihood(self) -> float:
        return float(np.logpdf(x=self.y, cov=self.S))

    def mahalanobis(self) -> float:
        return float(np.sqrt(np.linalg.multi_dot([self.y.T, np.linalg.inv(self.S), self.y])))


class AdaptiveKalmanFilter(KalmanFilter):
    def __init__(self, dx: int, dz: int, dt: float = 1, measurement_std: float = 1, process_std: float = 1,
                 alpha: float = 0.3):
        super().__init__(dx=dx, dz=dz, dt=dt, measurement_std=measurement_std, process_std=process_std)
        self.alpha = alpha

    def update(self, z: Sequence) -> np.array:
        if not self.initialized:
            raise ValueError('KalmanFilter not initialized. Please call `initialize(x: Sequence, std: float)` before '
                             'calling `predict()` or `update()` methods')

        z = np.asarray(z)
        z = np.expand_dims(z, axis=1) if z.ndim == 1 else z

        PHT = np.dot(self.P, self.H.T)
        HPHT = np.dot(self.H, PHT)
        self.S = HPHT + self.R
        self.K = np.dot(PHT, np.linalg.inv(self.S))

        self.y = z - np.dot(self.H, self.x)
        self.x += np.dot(self.K, self.y)
        y = z - np.dot(self.H, self.x)
        IKH = self._Ix - np.dot(self.K, self.H)
        self.P = np.linalg.multi_dot([IKH, self.P, IKH.T]) + np.linalg.multi_dot([self.K, self.R, self.K.T])
        self.Q = self.alpha * self.Q + (1 - self.alpha) * np.linalg.multi_dot([self.K, self.y, self.y.T, self.K.T])
        self.R = self.alpha * self.R + (1 - self.alpha) * (np.dot(y, y.T) + HPHT)
        return self.x


class KalmanFilterConstantVelocity(KalmanFilter):
    def __init__(self, dx: int, dz: int, dt: float = 1, measurement_std: float = 1, process_std: float = 1,
                 alpha: float = 0.3):
        super().__init__(dx=dx, dz=dz, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)
        self.F = np.kron(self._Iz, [[1, dt], [0, 1]])
        self.G = np.kron(self._Iz, [dt ** 2 / 2, dt])
        self.H = np.kron(self._Iz, [1, 0])
        self.Q = np.linalg.multi_dot([self.G.T, self._Iz, self.G]) * process_std ** 2

    def __str__(self) -> str:
        return f"KalmanFilter[x_dimensions={self.dx}, z_dimensions={self.dz}, motion='ConstantVelocity']"


class KalmanFilterConstantVelocity1D(KalmanFilterConstantVelocity):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=2, dz=1, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantVelocity2D(KalmanFilterConstantVelocity):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=4, dz=2, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantVelocity3D(KalmanFilterConstantVelocity):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=6, dz=3, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantVelocity4D(KalmanFilterConstantVelocity):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=8, dz=4, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantAcceleration(KalmanFilter):
    def __init__(self, dx: int, dz: int, dt: float = 1, measurement_std: float = 1, process_std: float = 1,
                 alpha: float = 0.3):
        super().__init__(dx=dx, dz=dz, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)
        self.F = np.kron(self._Iz, [[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
        self.G = np.kron(self._Iz, [dt ** 2 / 2, dt, 1])
        self.H = np.kron(self._Iz, [1, 0, 0])
        self.Q = np.linalg.multi_dot([self.G.T, self._Iz, self.G]) * process_std ** 2

    def __str__(self) -> str:
        return f"KalmanFilter[x_dimensions={self.dx}, z_dimensions={self.dz}, motion='ConstantAcceleration']"


class KalmanFilterConstantAcceleration1D(KalmanFilterConstantAcceleration):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=3, dz=1, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantAcceleration2D(KalmanFilterConstantAcceleration):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=6, dz=2, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantAcceleration3D(KalmanFilterConstantAcceleration):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=9, dz=3, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)


class KalmanFilterConstantAcceleration4D(KalmanFilterConstantAcceleration):
    def __init__(self, dt: float = 1, measurement_std: float = 1, process_std: float = 1, alpha: float = 0.3):
        super().__init__(dx=12, dz=4, dt=dt, measurement_std=measurement_std, process_std=process_std, alpha=alpha)

import zarr
import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from torch_geometric.data import Data, Batch

import torch


class DataAPI:
    """
    Оптимизированный API для хранилища данных с улучшенной производительностью
    """

    # Константы для имен массивов
    SPARSE_DATA = "sparse/data"
    SPARSE_ROW = "sparse/row"
    SPARSE_COL = "sparse/col"
    DENSE_DATA = "dense/data"
    LABELS = "labels"
    NAVIGATION_INDICES = "navigation"

    # Оптимизированные настройки сжатия (меньший уровень для скорости)
    DATA_COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    INDEX_COMPRESSOR = zarr.Blosc(cname="lz4", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE)

    def __init__(
        self,
        path: str,
        group_shape: Tuple[int, int, int],
        mode: str = "a",
        chunk_size: int = 1024 * 1024,  # 1MB чанки
        buffer_size: int = 3000,
        max_workers: int = 8,
    ):
        """
        Инициализация хранилища с оптимизированными параметрами.

        Args:
            path: Путь к хранилищу Zarr
            group_shape: Форма группы (количество матриц, строк, столбцов)
            mode: Режим доступа ('a' - чтение/запись, 'r' - только чтение, 'w' - перезапись)
            chunk_size: Размер чанков для хранения данных
            buffer_size: Количество групп в буфере перед записью на диск
            max_workers: Количество потоков для параллельных операций
        """
        self.path = path
        self.mode = mode.lower()
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self._group_shape = group_shape
        self.max_workers = max_workers
        self._validate_init_params()

        # Инициализация хранилища
        self.store = zarr.DirectoryStore(path)
        self.exists = os.path.exists(os.path.join(path, ".zgroup"))
        self._initialize_or_open_storage()
        self._reset_buffers()

    def _validate_init_params(self):
        """Проверка параметров инициализации."""
        if self.mode not in ("a", "r", "w"):
            raise ValueError("Режим должен быть 'a', 'r' или 'w'")
        if self.chunk_size <= 0:
            raise ValueError("Размер чанка должен быть положительным")
        if self.buffer_size <= 0:
            raise ValueError("Размер буфера должен быть положительным")
        if any(dim <= 0 for dim in self._group_shape):
            raise ValueError("Все размеры группы должны быть положительными")

    def _reset_buffers(self):
        """Сброс буферов записи с предварительным выделением памяти."""
        # Предварительное выделение памяти для буферов
        max_sparse_elements = (
            self.buffer_size * self._group_shape[0] * self._group_shape[1] ** 2
        )
        max_dense_elements = (
            self.buffer_size
            * self._group_shape[0]
            * self._group_shape[1]
            * self._group_shape[2]
        )

        self._sparse_buffer = {
            "data": np.empty(max_sparse_elements, dtype=np.float32),
            "row": np.empty(max_sparse_elements, dtype=np.int32),
            "col": np.empty(max_sparse_elements, dtype=np.int32),
            "pos": 0,
        }

        self._dense_buffer = {
            "data": np.empty(max_dense_elements, dtype=np.float32),
            "pos": 0,
        }

        self._label_buffer = np.empty(self.buffer_size, dtype=np.int32)
        self._navigation_buffer = np.empty(
            self.buffer_size * self._group_shape[0], dtype=np.int64
        )
        self._buffered_groups = 0

    def _initialize_or_open_storage(self):
        """Инициализация нового или открытие существующего хранилища."""
        if self.mode == "w":
            self._initialize_new_store()
        elif self.mode == "a":
            if self.exists:
                self._open_existing_store()
            else:
                self._initialize_new_store()
        elif self.mode == "r":
            if not self.exists:
                raise FileNotFoundError(
                    f"Zarr хранилище не найдено по пути {self.path}"
                )
            self._open_existing_store()

    def _initialize_new_store(self):
        """Инициализация нового хранилища с оптимизированными параметрами."""
        self.root = zarr.group(store=self.store, overwrite=True)

        # Разреженные матрицы (COO формат)
        self.root.create_dataset(
            self.SPARSE_DATA,
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype=np.float32,
            compressor=self.DATA_COMPRESSOR,
            write_empty_chunks=False,
            fill_value=0,
        )
        self.root.create_dataset(
            self.SPARSE_ROW,
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype=np.int32,
            compressor=self.INDEX_COMPRESSOR,
            write_empty_chunks=False,
            fill_value=0,
        )
        self.root.create_dataset(
            self.SPARSE_COL,
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype=np.int32,
            compressor=self.INDEX_COMPRESSOR,
            write_empty_chunks=False,
            fill_value=0,
        )

        # Плотные матрицы
        self.root.create_dataset(
            self.DENSE_DATA,
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype=np.float32,
            compressor=self.DATA_COMPRESSOR,
            write_empty_chunks=False,
            fill_value=0,
        )

        # Метки
        self.root.create_dataset(
            self.LABELS,
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype=np.int32,
            compressor=self.INDEX_COMPRESSOR,
            write_empty_chunks=False,
            fill_value=-1,
        )

        # Навигационные индексы
        self.root.create_dataset(
            self.NAVIGATION_INDICES,
            shape=(1,),
            chunks=(self.chunk_size,),
            dtype=np.int64,
            compressor=self.INDEX_COMPRESSOR,
            write_empty_chunks=False,
            fill_value=0,
        )
        self.root[self.NAVIGATION_INDICES][0] = 0

        # Метаданные
        self.root.attrs.update(
            {
                "version": "2.0",
                "created": np.datetime64("now").astype(str),
                "next_group_id": 0,
                "group_shape": self._group_shape,
            }
        )

    def _open_existing_store(self):
        """Открытие существующего хранилища."""
        self.root = zarr.group(store=self.store)
        # Проверка совместимости формы группы
        stored_shape = tuple(self.root.attrs.get("group_shape", (0, 0, 0)))
        if stored_shape != self._group_shape:
            warnings.warn(
                f"Форма группы в хранилище {stored_shape} не соответствует ожидаемой {self._group_shape}",
                RuntimeWarning,
            )

    def add_group(
        self,
        sparse_matrices: Optional[List[sp.coo_matrix]] = None,
        dense_matrices: Optional[List[np.ndarray]] = None,
        label: Optional[int] = None,
    ):
        """
        Добавление новой группы матриц в хранилище.

        Args:
            sparse_matrices: Список разреженных матриц (COO)
            dense_matrices: Список плотных матриц
            label: Метка группы
        """
        if sparse_matrices is not None:
            self._process_sparse_matrices(sparse_matrices)

        if dense_matrices is not None:
            self._process_dense_matrices(dense_matrices)

        self._label_buffer[self._buffered_groups] = label if label is not None else -1
        self._buffered_groups += 1

        if self._buffered_groups >= self.buffer_size:
            self.flush()

    def _process_sparse_matrices(self, sparse_matrices: List[sp.coo_matrix]):
        """Оптимизированная обработка разреженных матриц в формате COO."""
        if len(sparse_matrices) != self._group_shape[0]:
            raise ValueError(f"Ожидается {self._group_shape[0]} матриц в группе")

        for i, mat_coo in enumerate(sparse_matrices):
            data_len = mat_coo.data.size
            start = self._sparse_buffer["pos"]
            end = start + data_len

            # Заполнение буферов
            self._sparse_buffer["data"][start:end] = mat_coo.data
            self._sparse_buffer["row"][start:end] = mat_coo.row
            self._sparse_buffer["col"][start:end] = mat_coo.col
            self._sparse_buffer["pos"] = end

            # Навигационные данные
            nav_pos = self._buffered_groups * self._group_shape[0] + i
            self._navigation_buffer[nav_pos] = data_len

    def _process_dense_matrices(self, matrices: List[np.ndarray]):
        """Оптимизированная обработка плотных матриц."""
        if len(matrices) != self._group_shape[0]:
            raise ValueError(f"Ожидается {self._group_shape[0]} матриц в группе")

        for mat in matrices:
            if mat.shape != (self._group_shape[1], self._group_shape[2]):
                raise ValueError(
                    f"Неверная форма матрицы: ожидается {(self._group_shape[1], self._group_shape[2])}"
                )

            start = self._dense_buffer["pos"]
            end = start + mat.size
            self._dense_buffer["data"][start:end] = mat.ravel()
            self._dense_buffer["pos"] = end

    def flush(self):
        """Запись буферизированных данных на диск с оптимизацией."""
        if self._buffered_groups == 0:
            return

        try:
            # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            #     # Параллельная запись разных частей данных
            #     futures = []

            #     if self._sparse_buffer["pos"] > 0:
            #         futures.append(executor.submit(self._write_sparse_data))

            #     if self._dense_buffer["pos"] > 0:
            #         futures.append(executor.submit(self._write_dense_data))

            #     futures.append(executor.submit(self._write_labels))

            #     # Ожидание завершения всех операций
            #     for future in as_completed(futures):
            #         future.result()
            
            if self._sparse_buffer["pos"] > 0:
                self._write_sparse_data()
                
            if self._dense_buffer["pos"] > 0:
                self._write_dense_data()
                
            self._write_labels()

        except Exception as e:
            self._reset_buffers()
            raise RuntimeError(f"Ошибка при записи буфера: {str(e)}") from e

        self._reset_buffers()

    def _write_sparse_data(self):
        """Оптимизированная запись разреженных данных в формате COO."""
        # Запись data, row и col
        current_size = self.root[self.SPARSE_DATA].shape[0]
        new_size = current_size + self._sparse_buffer["pos"]

        for array_name in [self.SPARSE_DATA, self.SPARSE_ROW, self.SPARSE_COL]:
            self.root[array_name].resize(new_size)

        self.root[self.SPARSE_DATA][current_size:new_size] = self._sparse_buffer[
            "data"
        ][: self._sparse_buffer["pos"]]
        self.root[self.SPARSE_ROW][current_size:new_size] = self._sparse_buffer["row"][
            : self._sparse_buffer["pos"]
        ]
        self.root[self.SPARSE_COL][current_size:new_size] = self._sparse_buffer["col"][
            : self._sparse_buffer["pos"]
        ]

        # Обновление навигационных индексов
        prev_nav_size = self.root[self.NAVIGATION_INDICES].shape[0]
        new_nav_size = prev_nav_size + self._buffered_groups * self._group_shape[0]

        self.root[self.NAVIGATION_INDICES].resize(new_nav_size)
        cumsum = np.cumsum(
            self._navigation_buffer[: self._buffered_groups * self._group_shape[0]]
        )
        self.root[self.NAVIGATION_INDICES][prev_nav_size:new_nav_size] = (
            cumsum + self.root[self.NAVIGATION_INDICES][-1]
        )

    def _write_dense_data(self):
        """Оптимизированная запись плотных данных."""
        current_size = self.root[self.DENSE_DATA].shape[0]
        new_size = current_size + self._dense_buffer["pos"]

        self.root[self.DENSE_DATA].resize(new_size)
        self.root[self.DENSE_DATA][current_size:new_size] = self._dense_buffer["data"][
            : self._dense_buffer["pos"]
        ]

    def _write_labels(self):
        """Запись меток."""
        current_size = self.root[self.LABELS].shape[0]
        new_size = current_size + self._buffered_groups

        self.root[self.LABELS].resize(new_size)
        self.root[self.LABELS][current_size:new_size] = self._label_buffer[:self._buffered_groups]

    def get_sample(
        self, group_id: int
    ) -> Tuple[List[Tuple[sp.coo_matrix, np.ndarray]], int]:
        """
        Оптимизированное чтение группы данных.

        Args:
            group_id: ID группы для чтения

        Returns:
            Кортеж (список матриц, метка)
        """
        return self._get_sample_optimized(group_id)

    def _get_sample_optimized(
        self, group_id: int
    ) -> Tuple[List[Tuple[sp.coo_matrix, np.ndarray]], int]:
        """Векторизованная реализация чтения группы данных."""
        # Чтение навигационных индексов
        nav_start = group_id * self._group_shape[0]
        nav_end = (group_id + 1) * self._group_shape[0] + 1
        group_offsets = self.root[self.NAVIGATION_INDICES][nav_start:nav_end]

        # Чтение разреженных данных (COO формат)
        sparse_data = self.root[self.SPARSE_DATA][group_offsets[0] : group_offsets[-1]]
        sparse_row = self.root[self.SPARSE_ROW][group_offsets[0] : group_offsets[-1]]
        sparse_col = self.root[self.SPARSE_COL][group_offsets[0] : group_offsets[-1]]

        # Чтение плотных данных
        dense_start = (
            group_id
            * self._group_shape[0]
            * self._group_shape[1]
            * self._group_shape[2]
        )
        dense_end = (
            (group_id + 1)
            * self._group_shape[0]
            * self._group_shape[1]
            * self._group_shape[2]
        )
        dense_data = self.root[self.DENSE_DATA][dense_start:dense_end]

        # Чтение метки
        label = self.root[self.LABELS][group_id]

        # Создание списка матриц
        matrices = []
        data_sizes = np.diff(group_offsets)
        data_offsets = np.concatenate(([0], np.cumsum(data_sizes)))

        for i in range(self._group_shape[0]):
            # Создание разреженной матрицы (COO)
            data_slice = slice(data_offsets[i], data_offsets[i + 1])

            # Создание плотной матрицы
            dense_slice = slice(
                i * self._group_shape[1] * self._group_shape[2],
                (i + 1) * self._group_shape[1] * self._group_shape[2],
            )
            dense_mat = torch.tensor(
                dense_data[dense_slice].reshape(
                    self._group_shape[1], self._group_shape[2]
                )
            )

            edge_index = torch.tensor(
                np.array(
                    [sparse_row[data_slice], sparse_col[data_slice]], dtype=np.int64
                )
            )

            edge_attr = torch.tensor(sparse_data[data_slice], dtype=torch.float32)

            graph = Data(x=dense_mat, edge_index=edge_index, edge_attr=edge_attr)

            matrices.append(graph)

        return matrices, torch.tensor(label)

    @contextmanager
    def batch_writer(self):
        """Контекстный менеджер для пакетной записи."""
        try:
            yield self
        finally:
            self.flush()

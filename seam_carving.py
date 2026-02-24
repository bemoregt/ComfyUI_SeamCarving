"""
Seam Carving algorithm implementation using pure NumPy.

Seam Carving (Avidan & Shamir, 2007) – 이미지의 중요도가 낮은 경로(seam)를
제거하거나 삽입해 콘텐츠 손상 없이 비선형 리사이징을 수행합니다.

지원 동작:
  - 폭/높이 축소: 최소 에너지 seam 순차 제거
  - 폭/높이 확대: k개 seam 동시 탐색 후 삽입 (단순 반복 삽입의 줄무늬 방지)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Energy map
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    """H×W×C → H×W 그레이스케일 (float64)."""
    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.float64)
    # Rec.601 luminance
    return (0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]).astype(np.float64)


def compute_energy(image: np.ndarray) -> np.ndarray:
    """
    중앙차분 그래디언트로 에너지 맵을 계산합니다.

    Parameters
    ----------
    image : ndarray, shape (H, W, C), float32/64, 범위 [0, 1]

    Returns
    -------
    energy : ndarray, shape (H, W), float64
    """
    gray = _to_gray(image)
    # edge 패딩 후 중앙차분 → 경계에서도 왜곡 최소화
    padded = np.pad(gray, 1, mode="edge")
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
    return np.sqrt(gx ** 2 + gy ** 2)


# ---------------------------------------------------------------------------
# 2. DP seam finding
# ---------------------------------------------------------------------------

def _find_seam_dp(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    수직 seam DP.

    dp[i, j]        = (i, j)까지의 누적 최소 에너지
    backtrack[i, j] = (i, j)로 오기 위해 선택한 row i-1 의 column 인덱스

    경계는 inf 패딩으로 자연스럽게 처리합니다.
    """
    H, W = energy.shape
    dp = energy.astype(np.float64)          # 첫 행은 energy 그대로
    backtrack = np.empty((H, W), dtype=np.int32)
    backtrack[0] = np.arange(W)             # 첫 행은 자기 자신

    col_idx = np.arange(W)

    for i in range(1, H):
        # inf 패딩으로 경계를 자동 처리 (좌우 경계에서 벽 방향 선택 불가)
        padded = np.pad(dp[i - 1], (1, 1), constant_values=np.inf)
        left   = padded[:-2]    # dp[i-1, j-1]  j=0이면 inf
        center = padded[1:-1]   # dp[i-1, j]
        right  = padded[2:]     # dp[i-1, j+1]  j=W-1이면 inf

        choices = np.stack([left, center, right])   # (3, W)
        best    = np.argmin(choices, axis=0)         # 0=왼쪽, 1=중앙, 2=오른쪽

        backtrack[i] = col_idx + best - 1           # 실제 이전 column 인덱스
        dp[i] += choices[best, col_idx]

    return dp, backtrack


def _trace_seam(dp: np.ndarray, backtrack: np.ndarray) -> np.ndarray:
    """DP 결과로부터 seam(column 인덱스 배열, 길이 H)을 역추적합니다."""
    H, W = dp.shape
    seam = np.empty(H, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for i in range(H - 2, -1, -1):
        prev = backtrack[i + 1, seam[i + 1]]
        seam[i] = int(np.clip(prev, 0, W - 1))
    return seam


# ---------------------------------------------------------------------------
# 3. Seam removal / insertion primitives
# ---------------------------------------------------------------------------

def _remove_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """수직 seam 1개를 제거합니다. (H, W, C) → (H, W-1, C)"""
    H, W, C = image.shape
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False
    return image[mask].reshape(H, W - 1, C)


def _collect_seams(image: np.ndarray, k: int) -> list[np.ndarray]:
    """
    원본 이미지에서 k개의 seam을 순차적으로 찾아 '원본 좌표계'로 반환합니다.

    col_map을 이용해 축소된 임시 이미지의 column → 원본 column 변환을 추적합니다.
    반환된 seam 리스트는 이후 원본 이미지에 한 번에 삽입할 때 사용됩니다.
    """
    H, W, C = image.shape
    temp    = image.copy()
    col_map = np.tile(np.arange(W, dtype=np.int32), (H, 1))  # 현재→원본 column 맵
    seam_originals: list[np.ndarray] = []

    for _ in range(k):
        energy = compute_energy(temp)
        dp, bt = _find_seam_dp(energy)
        seam   = _trace_seam(dp, bt)

        # 현재 seam의 column을 원본 좌표로 변환
        seam_originals.append(col_map[np.arange(H), seam].copy())

        # 임시 이미지·col_map에서 seam 제거
        cur_W = temp.shape[1]
        mask  = np.ones((H, cur_W), dtype=bool)
        mask[np.arange(H), seam] = False
        temp    = temp[mask].reshape(H, cur_W - 1, C)
        col_map = col_map[mask].reshape(H, cur_W - 1)

    return seam_originals


def _insert_seams(image: np.ndarray, seam_list: list[np.ndarray]) -> np.ndarray:
    """
    원본 이미지에 seam_list 에 담긴 seam들을 한 번에 삽입합니다.

    각 삽입 픽셀은 왼쪽·현재 픽셀의 평균으로 채웁니다.
    """
    H, W, C = image.shape
    k       = len(seam_list)
    img_f   = image.astype(np.float64)
    output  = np.empty((H, W + k, C), dtype=np.float64)

    for row in range(H):
        # 해당 행의 삽입 위치(원본 좌표)를 정렬
        ins_cols = sorted(s[row] for s in seam_list)
        new_row: list[np.ndarray] = []
        ptr = 0  # ins_cols 포인터

        for col in range(W):
            # col 위치에 삽입할 픽셀이 있으면 먼저 삽입
            while ptr < len(ins_cols) and ins_cols[ptr] == col:
                left = img_f[row, col - 1] if col > 0 else img_f[row, col]
                new_row.append((left + img_f[row, col]) * 0.5)
                ptr += 1
            new_row.append(img_f[row, col])

        # 끝에 남은 삽입 (이론상 발생하지 않지만 안전 처리)
        while ptr < len(ins_cols):
            new_row.append(img_f[row, W - 1])
            ptr += 1

        output[row] = np.array(new_row[:W + k], dtype=np.float64)

    return np.clip(output, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Width / Height adjustment
# ---------------------------------------------------------------------------

def _carve_width(
    image: np.ndarray,
    target_w: int,
    step_cb: "Callable[[], None] | None" = None,
) -> np.ndarray:
    """이미지 폭을 target_w 로 조정합니다."""
    W = image.shape[1]
    delta = target_w - W

    if delta == 0:
        return image

    if delta < 0:
        # ---- 축소: seam 순차 제거 ----
        img = image.copy()
        for _ in range(-delta):
            energy = compute_energy(img)
            dp, bt = _find_seam_dp(energy)
            seam   = _trace_seam(dp, bt)
            img    = _remove_seam(img, seam)
            if step_cb:
                step_cb()
        return img

    else:
        # ---- 확대: k개 seam 탐색 후 일괄 삽입 ----
        seam_list = _collect_seams(image, delta)
        result    = _insert_seams(image, seam_list)
        if step_cb:
            for _ in range(delta):
                step_cb()
        return result


def _carve_height(
    image: np.ndarray,
    target_h: int,
    step_cb: "Callable[[], None] | None" = None,
) -> np.ndarray:
    """이미지 높이를 target_h 로 조정합니다 (전치 후 폭 조정, 복원)."""
    # (H, W, C) → (W, H, C) : 수직 seam = 수평 seam
    img_t  = image.transpose(1, 0, 2)
    res_t  = _carve_width(img_t, target_h, step_cb)
    return res_t.transpose(1, 0, 2)


# ---------------------------------------------------------------------------
# 5. Public API
# ---------------------------------------------------------------------------

def seam_carve(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    step_cb: "Callable[[], None] | None" = None,
) -> np.ndarray:
    """
    Seam Carving으로 이미지를 (target_height, target_width, C) 크기로 리사이징합니다.

    Parameters
    ----------
    image        : ndarray (H, W, C), float32, 값 범위 [0, 1]
    target_width : 출력 폭  (1 이상)
    target_height: 출력 높이 (1 이상)
    step_cb      : seam 1개 처리 완료마다 호출되는 콜백 (프로그레스 바용)

    Returns
    -------
    ndarray (target_height, target_width, C), float32, 값 범위 [0, 1]
    """
    img = image.astype(np.float32)
    img = _carve_width(img, target_width, step_cb)
    img = _carve_height(img, target_height, step_cb)
    return img


def total_steps(src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> int:
    """예상 총 seam 처리 횟수를 반환합니다 (프로그레스 바 초기화용)."""
    return abs(tgt_w - src_w) + abs(tgt_h - src_h)

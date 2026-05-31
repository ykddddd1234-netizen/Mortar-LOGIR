import numpy as np
from scipy.interpolate import CubicSpline

# =========================================
# Baseline Spline Nodes (아음속 제어점)
# =========================================
# 장약 6번의 절벽 구간(M=0.93~)에 간섭하지 않도록 M=0.90까지만 노드를 배치합니다.
BASE_MACH_NODES = np.array([
    0.0,    # Node 0
    0.65,   # Node 1 (장약 3)
    0.71,   # Node 2 (장약 4)
    0.77,   # Node 3 (장약 4)
    0.805,  # Node 4 (천음속 진입 완충)
    0.83,   # Node 5 (장약 5 고도 정밀 제어선) ★ 추가
    0.85,   # Node 6 (장약 5 사거리 정밀 제어선) ★ 추가
    0.87,   # Node 7 (장약 6 진입 완충) ★ 추가
    0.89,   # Node 8 (장약 6)
    0.94    # Node 9
])

# =========================================
# Hump Function (천음속 드래그 라이즈)
# =========================================
def hump(M, params):
    """
    천음속 영역의 급격한 항력 장벽(Hump)을 계산합니다.
    장약 6번의 기온 변동성을 기막히게 맞춘 심장부이므로 기존 공식과 파라미터를 그대로 유지합니다.
    """
    A = params["A"]
    M1 = params["M1"]
    M2 = params["M2"]
    k1 = params["k1"]
    k2 = params["k2"]

    rise = 1.0 / (1.0 + np.exp(-k1 * (M - M1)))
    fall = 1.0 / (1.0 + np.exp(k2 * (M - M2)))

    return A * rise * fall


# =========================================
# Total Cd (Baseline Spline + Hump)
# =========================================
def get_cd(M, params):
    """
    스플라인 기반의 아음속 Baseline 항력과 천음속 Hump 항력을 더해 전체 Cd를 구합니다.
    """
    # 1. params 내부에 CubicSpline 객체가 이미 만들어져 있다면 그것을 재사용 (연산 속도 최적화)
    if "cs_object" in params:
        cs = params["cs_object"]
    else:
        # 안전장치: 오브젝트가 없을 때만 루프 외부에서 생성
        cs = CubicSpline(BASE_MACH_NODES, params["baseline_cds"], bc_type="clamped")
    
    # 2. Baseline Cd 결정 (노드 경계 밖 예외 처리로 외삽 발산 방지)
    if M > BASE_MACH_NODES[-1]:
        base = float(cs(BASE_MACH_NODES[-1]))
    elif M < BASE_MACH_NODES[0]:
        base = float(cs(BASE_MACH_NODES[0]))
    else:
        base = float(cs(M))
        
    # 3. Baseline 항력과 독립된 Hump 항력의 결합
    return base + hump(M, params)
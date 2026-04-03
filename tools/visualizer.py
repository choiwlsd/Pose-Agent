def print_features(features):
    # None이 아닌 feature만 출력
    if features is None:
        return

    wtsl = features['wrist_to_shoulder_left']
    ew   = features['elbow_width']
    wvl  = features['wrist_velocity_left']
    ea   = features['elbow_angle']
    war  = features['wrist_angle_right']

    print("=" * 40)
    if wtsl:             print(f"  왼손목-왼어깨 거리:      {wtsl:.3f}")
    if ew:               print(f"  양쪽 팔꿈치 거리:        {ew:.3f}")
    if wvl:              print(f"  왼손목 속도:             {wvl:.4f}")
    if 'left' in ea:     print(f"  왼팔 각도:               {ea['left']:.1f}°")
    if 'right' in ea:    print(f"  오른팔 각도:             {ea['right']:.1f}°")
    if war:              print(f"  오른손목 각도:           {war:.1f}°")
    print("=" * 40)
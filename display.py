def print_features(features):
    # None이면 출력하지 않음 
    if features is None:
        return

    wts = features['wrist_to_shoulder']
    sw  = features['shoulder_width']
    wv  = features['wrist_velocity']
    ea  = features['elbow_angle']
    wa  = features['wrist_angle']

    print("=" * 40)
    if 'left' in wts:  print(f"  왼손목-왼어깨 거리:      {wts['left']:.3f}")
    if 'right' in wts: print(f"  오른손목-오른어깨 거리:   {wts['right']:.3f}")
    if sw:             print(f"  어깨 너비:               {sw:.3f}")
    if wv:
        if 'left' in wv:   print(f"  왼손목 속도:             {wv['left']:.4f}")
        if 'right' in wv:  print(f"  오른손목 속도:           {wv['right']:.4f}")
    if 'left' in ea:   print(f"  왼팔꿈치 각도:           {ea['left']:.1f}°")
    if 'right' in ea:  print(f"  오른팔꿈치 각도:         {ea['right']:.1f}°")
    if 'left' in wa:   print(f"  왼손목 각도:             {wa['left']:.1f}°")
    if 'right' in wa:  print(f"  오른손목 각도:           {wa['right']:.1f}°")
    print("=" * 40)
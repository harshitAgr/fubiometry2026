from experiments import groups as g

def test_cardiac_dcm_clip_groups_frames_together():
    a = g.group_key("A4C", "A4C/DCM_IM_0008_frame031.png")
    b = g.group_key("A4C", "A4C/DCM_IM_0008_frame061.png")
    c = g.group_key("A4C", "A4C/DCM_IM_0009_frame002.png")
    assert a == b and a != c

def test_cardiac_series_is_distinct():
    base = g.group_key("PLAX", "PLAX/DCM_IM_0008_frame031.png")
    ser = g.group_key("PLAX", "PLAX/DCM_IM_0008_s2_frame031.png")
    assert base != ser

def test_cardiac_png_family_is_grouped_not_per_image():
    a = g.group_key("PSAX", "PSAX/PNG_IM_12-3.png")
    b = g.group_key("PSAX", "PSAX/PNG_IM_12-3.png")
    c = g.group_key("PSAX", "PSAX/PNG_IM_99.png")
    assert a == b and a != c
    assert "img" not in a

def test_fetal_femur_groups_by_patient():
    a = g.group_key("fetal_femur", "fetal_femur/Patient00168_Plane5_1_of_2.png")
    b = g.group_key("fetal_femur", "fetal_femur/Patient00168_Plane5_2_of_2.png")
    c = g.group_key("fetal_femur", "fetal_femur/Patient00627_Plane5_1_of_1.png")
    assert a == b and a != c

def test_obstetric_is_per_image():
    a = g.group_key("HC", "HC/000_HC.png")
    b = g.group_key("HC", "HC/001_HC.png")
    assert a != b

def test_aop_frame_index():
    assert g.aop_frame_index("AOP/00042.jpg") == 42
    assert g.aop_frame_index("AOP/04000.jpg") == 4000

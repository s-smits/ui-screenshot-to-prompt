from ui_screenshot_to_prompt import config
from ui_screenshot_to_prompt import detection


def test_create_detection_qualifies_type():
    detection_term = config.get_detection_term()
    result = detection.DetectorBase.create_detection((0, 0, 10, 10), detection_type="button")
    assert result.type == f"{detection_term}_button"

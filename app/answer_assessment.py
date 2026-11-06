"""Server-localized limitation messages; model reports are never source evidence."""

from app.contracts import AnswerAssessment, AnswerLimitation

MESSAGES = {
    "en": {
        "MISSING_SUPPLIED_EVIDENCE": (
            "The supplied passages do not establish this part of the question.",
            "Find a documentation passage covering the missing part; absence here does not prove absence in the corpus."),
        "ACCOUNT_STATE_UNAVAILABLE": (
            "This documentation-only mode cannot verify the current state of an AWS account.",
            "Check the relevant configuration in the AWS Console; this mode has no account connection."),
        "SOURCE_SCOPE_AMBIGUITY": (
            "The supplied evidence does not establish the required scope or conditions.",
            "Clarify the service variant, version, region or account scope and find matching documentation."),
        "CONFLICTING_SUPPLIED_EVIDENCE": (
            "The supplied passages contain an unresolved evidence conflict.",
            "Check the document versions and applicable conditions before relying on a conclusion."),
        "OUT_OF_SCOPE": (
            "The reported question part is outside the supplied documentation's scope.",
            "Ask about the supplied documentation or find an appropriate documentation source."),
    },
    "vi": {
        "MISSING_SUPPLIED_EVIDENCE": (
            "Các đoạn tài liệu được cấp chưa chứng minh phần này của câu hỏi.",
            "Tìm đoạn tài liệu hỗ trợ phần còn thiếu; thiếu trong context không có nghĩa toàn corpus không có."),
        "ACCOUNT_STATE_UNAVAILABLE": (
            "Chế độ chỉ đọc tài liệu không thể xác minh trạng thái hiện tại của tài khoản AWS.",
            "Kiểm tra cấu hình liên quan trong AWS Console; chế độ này không kết nối tài khoản của bạn."),
        "SOURCE_SCOPE_AMBIGUITY": (
            "Bằng chứng được cấp chưa xác định đủ phạm vi hoặc điều kiện áp dụng.",
            "Làm rõ biến thể dịch vụ, phiên bản, region hoặc phạm vi tài khoản rồi tìm tài liệu tương ứng."),
        "CONFLICTING_SUPPLIED_EVIDENCE": (
            "Các đoạn nguồn được cấp có mâu thuẫn bằng chứng chưa được giải quyết.",
            "Kiểm tra phiên bản tài liệu và điều kiện áp dụng trước khi dựa vào kết luận."),
        "OUT_OF_SCOPE": (
            "Phần câu hỏi được nêu nằm ngoài phạm vi tài liệu được cấp.",
            "Hỏi về tài liệu được cấp hoặc tìm nguồn tài liệu phù hợp với phần đó."),
    },
}


def build_assessment(gaps, *, language, reported_by="MODEL_REPORTED"):
    """Only enum-selected templates are trusted; question_part remains model text."""
    if language not in MESSAGES or reported_by not in {"MODEL_REPORTED", "SYSTEM_POLICY"}:
        raise ValueError("ASSESSMENT_LANGUAGE_OR_ORIGIN_INVALID")
    limitations = []
    for gap in gaps:
        message, next_step = MESSAGES[language][gap.code]
        limitations.append(AnswerLimitation(code=gap.code, reported_by=reported_by,
            question_part=gap.question_part if reported_by == "MODEL_REPORTED" else None,
            message=message, next_step=next_step))
    return AnswerAssessment(limitations=tuple(limitations))

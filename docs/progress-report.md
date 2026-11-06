# Báo cáo tiến độ — AWS Evidence Lab

Cập nhật: 06/11/2026 · Lê Phú Quý · Dự án cá nhân

[Website AWS](https://cyg0uvml94.execute-api.us-east-1.amazonaws.com/) · [GitHub](https://github.com/LePhuQuy05/enterprise-knowledge-graph-assistant) · [Kiến trúc](architecture.md) · [Đánh giá](evaluation.md) · [Cách chạy](development.md)

## Sản phẩm đang phục vụ

Ứng dụng giúp người học AWS tìm bằng chứng, hiểu các luồng nhiều dịch vụ và kiểm tra nguồn của câu trả lời. AWS đảm nhiệm hosting, xác thực, hàng đợi và trạng thái; DeepSeek cung cấp generation, BGE ONNX tạo embeddings local. Hệ thống không đọc cấu hình hoặc log của tài khoản AWS của người hỏi.

| Phần | Đã triển khai | Phạm vi |
|---|---|---|
| Tài khoản | Username, email, mật khẩu website; xác minh email; đăng nhập | Cognito; không phải nút Google OAuth |
| Investigate | Trả lời Việt–Anh, citations, xem nguồn, history riêng và export brief | AWS official live search hoặc pilot snapshot; nguồn và phiên bản được tách riêng |
| Expanded collection V2 | 319 tài liệu / 3.024 chunks, tám phương pháp, compare tối đa ba phương pháp mỗi lượt | BM25, Vector, Hybrid, Reranker, Graph 1/2-hop, Multi-query Q, Query-balanced QB |
| Method pages | Sáu bước thực thi, nguồn, trace và map từ các facts đã chọn | Map thể hiện dữ liệu thực của run |
| UX | Dashboard, explorer, guided scenarios, onboarding; Câu hỏi mới giữ history | User xác nhận đổi câu và nhận kết quả mới ngày 06/11 |
| Lịch sử phiên bản | Recorded pilot 20 tài liệu / 236 chunks; legacy 316 tài liệu / 2.978 chunks | Không cộng các scope hoặc gán kết quả cũ thành kết quả 319 |

Graph cloud là snapshot có provenance, **27 facts từ 21 tài liệu**, chưa phủ toàn bộ 319. Neo4j chạy local; không có managed graph database chạy liên tục trên AWS. Câu hỏi Việt dùng shared English rewrite; Q/QB dùng cùng validated plan/pool với cách phân bổ riêng. Graph/plan không sẵn sàng giữ NOT_READY; không tự đổi sang Hybrid để làm như đã thành công.

## Bằng chứng kỹ thuật và giới hạn

- Managed AWS canary: 16 native answers trên tám phương pháp × hai ngôn ngữ; 21 model operations SETTLED. Published-rate upper bound 0,038708 USD, không phải hóa đơn. Chỉ một KMS intent đã được dùng trong development; không phải 16 câu độc lập hoặc accuracy 100%.
- Source audit: 184 source-byte checks, 135 citation appearances, tám selected graph clauses; không hidden fallback trong canary. Quote đúng vị trí không tự chứng minh claim đúng nghĩa.
- Worker V2 khớp 175 file Python trong deployed ZIP. Frontend khớp 119 public assets; [manifest](../configs/releases/collection319-v2.json) ghi hashes. Legacy query backend có package riêng; build current worker source không tạo lại exact historical query ZIP.
- Bản vá đổi câu: 159 Node checks; 14 focused, 51 V2 và 41 legacy browser checks; hai Python web tests; exact Linux tám checks / 119 routes. Đây là kết quả các đợt kiểm tương ứng, không phải answer accuracy hoặc tổng số tình huống độc lập.
- Backend có JWT/membership/owner checks, private S3, secret chỉ worker đọc, atomic quota/cost reservation và idempotency. Unknown completion giữ reservation; không tự gọi model lại bằng key mới. Đây là portfolio prototype, chưa phải chứng nhận security hoặc load testing.

## Mã nguồn và CI

Repo cung cấp source V2, public fixture preview và các bước kiểm tra trên Ubuntu trong [GitHub Actions](https://github.com/LePhuQuy05/enterprise-knowledge-graph-assistant/actions/workflows/ci.yml): cài dependencies khóa phiên bản, chuẩn bị tokenizer, Python tests, lint, Node tests, đối chiếu served-source pins, build preview và tổng hợp CDK. Xem kết quả của đúng commit đang đọc; không lấy lần CI cũ làm bằng chứng cho source mới. Các replay cần corpus/results local được báo skipped kèm lý do khi không có artifact.

## Phần còn lại

1. Review nhãn/nguồn và đánh giá chất lượng câu trả lời Việt–Anh trên recipe 319 hiện tại. Benchmark 100 tình huống trên 316 là diagnostic cũ có failures/fallback, không dùng để khẳng định Graph/Q/QB hiện tại tốt hơn Hybrid.
2. Cập nhật báo cáo/video/CV theo cùng release, rehearsal và kiểm hạn mức/giá trước buổi demo.

Dependencies được pin trong uv.lock và package-lock.json. Raw corpus, model weights, private history, private provider receipts, credentials và tài liệu học nội bộ không nằm trên GitHub. Các benchmark records đã được chọn lọc công khai vẫn được giữ riêng. Public fixture preview mô phỏng identity/provider, có nhãn và không gọi API trả phí; xem [development](development.md).

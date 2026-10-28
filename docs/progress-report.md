# Báo cáo tiến độ — AWS Evidence Lab

**Cập nhật: 28/10/2026 · Người thực hiện: Lê Phú Quý · Dự án cá nhân**

[Website AWS](https://cyg0uvml94.execute-api.us-east-1.amazonaws.com/) · [Mã nguồn](https://github.com/LePhuQuy05/enterprise-knowledge-graph-assistant) · [Kiến trúc](architecture.md) · [Đánh giá đã công bố](evaluation.md)

## 1. Kết quả hiện tại

Đã triển khai một ứng dụng web hỏi đáp tài liệu AWS với tài khoản riêng, câu trả lời Việt–Anh, trích dẫn có thể mở để kiểm tra, lịch sử và xuất bản tóm tắt. Một workspace riêng cho phép chạy và quan sát bốn phương pháp truy xuất trên 316 tài liệu AWS chính thức. Phần nghiên cứu GraphRAG, multi-query và query-balanced đang tiếp tục ở local trước khi đưa vào cùng luồng trên cloud.

Đây là báo cáo tiến độ một sản phẩm đã có bản chạy được, chưa phải tuyên bố hoàn thành toàn bộ kế hoạch hoặc trả lời đúng mọi câu hỏi AWS.

## 2. Các tính năng đã có trên website

Các hàng bên dưới có bằng chứng triển khai từ các đợt 18–25/10. Một số thao tác đăng nhập/hỏi đáp đã được người dùng xác nhận ở các đợt đó. Lần kiểm ngày 28/10 chỉ đối soát bề mặt public và tài nguyên giao diện, không chạy lại toàn bộ phiên đăng nhập hoặc gọi model.

| Trạng thái | Chức năng | Có thể trình diễn | Phạm vi / giới hạn |
|---|---|---|---|
| Đã triển khai | Đăng ký và đăng nhập | Username riêng, email/Gmail, mật khẩu website; xác minh email trước khi vào workspace | Dùng Cognito; không phải nút đăng nhập Google OAuth. Core account access đã được người dùng xác nhận |
| Đã triển khai | Luồng quên mật khẩu | Đi tới luồng khôi phục của Cognito | Chưa có xác nhận thử trọn luồng qua mailbox trong bộ nghiệm thu được đối soát |
| Đã triển khai | Hỏi đáp AWS bằng Việt–Anh | Investigate → AWS official · live search → chọn ngôn ngữ → hỏi về dịch vụ/luồng AWS | Tìm nguồn công khai khi hỏi; không phải đã index toàn bộ AWS. Không đọc cấu hình/log của tài khoản AWS |
| Đã triển khai | Hỏi đáp trên snapshot cố định | Chọn Pilot snapshot; truy xuất semantic hoặc hybrid | Pilot 20 tài liệu / 236 chunks, tách khỏi collection316 |
| Đã triển khai | Kiểm tra trích dẫn | Chọn statement/citation → đọc source passage → mở tài liệu AWS gốc | Vị trí/quote đúng không tự chứng minh toàn bộ ý nghĩa claim đúng |
| Đã triển khai | Lịch sử riêng theo tài khoản | Mở câu hỏi đã lưu, câu trả lời và nguồn | Mở lại bản ghi không gọi model mới; quyền đọc nguồn gắn với bản ghi thuộc người dùng |
| Đã triển khai | Guided scenarios và xuất brief | Chọn tình huống S3–Lambda/KMS, tạo câu hỏi có thể sửa; ghi notes và Export brief .md | Build câu hỏi chưa gửi model; notes trong phiên cần export để giữ lại |
| Đã triển khai | Dashboard và Knowledge Explorer | Overview, tìm tài liệu/chunks trong catalog, xem nguồn | Catalog/recorded dashboard hiện thuộc pilot20, không đại diện toàn corpus316/319 |
| Đã triển khai | Recorded Retrieval Lab | Xem thứ hạng, nhóm bằng chứng và kết quả graph thử nghiệm đã ghi | Dữ liệu phát lại, không chạy GraphRAG cho câu hỏi mới; nhãn đánh giá còn draft |
| Đã triển khai | Live Compare | Nhập câu hỏi → chọn BM25, Vector, Hybrid, Hybrid + Reranker → Run comparison | Chạy retrieval trên 316 tài liệu / 2.978 chunks / 24 nhóm dịch vụ; chưa sinh câu trả lời trên collection này |
| Đã triển khai | Theo dõi và giải thích run | Theo dõi tiến độ, mở nguồn, Load saved trace, Refresh runs, Export results .json | Run có quyền sở hữu; không cần submit lại để đọc kết quả |
| Đã triển khai | Quick tour | Popup hướng dẫn lần đầu; mở lại bằng Quick tour | Hướng dẫn thao tác, không tự gửi câu hỏi |

Các giới hạn vận hành đã được xây gồm JWT/owner isolation, idempotency, giới hạn tài nguyên truy xuất và cơ chế dự trù/ghi nhận chi phí model. Live Compare dùng queue và worker, không giữ một HTTP request chờ toàn bộ công việc. Các kiểm tra đã có chỉ bao phủ những tình huống được thử, không là chứng nhận an toàn toàn diện.

## 3. Ba phạm vi dữ liệu phải phân biệt

| Phạm vi | Quy mô | Được dùng ở đâu? | Trạng thái |
|---|---:|---|---|
| Recorded pilot | 20 tài liệu / 236 chunks | Snapshot answers, Knowledge Explorer, Recorded Lab | Đang có trên AWS |
| Collection316 | 316 tài liệu / 2.978 chunks / 24 nhóm | Live Compare, bốn phương pháp retrieval | Đang có trên AWS; chưa generation trên collection316 |
| Candidate319 | 319 tài liệu / 3.024 chunks | Nghiên cứu bổ sung nguồn, Graph/Q/QB, context và generation protocol mới | Local; chưa thay bản đang phục vụ |

AWS official · live search là một đường truy xuất khác: tìm và lưu bằng chứng theo câu hỏi, không có cùng quy mô hoặc điều kiện benchmark với các snapshot trên. Không cộng số trang archive đã tải vào số tài liệu đang được chatbot phục vụ.

## 4. Công nghệ và phần kỹ thuật có thể giải thích

| Phần | Công nghệ / cơ chế thực tế |
|---|---|
| Frontend | JavaScript modules, HTML/CSS; dashboard, source inspector, history và exports |
| Xác thực | Amazon Cognito, authorization code + PKCE, JWT |
| Backend AWS | API Gateway, Lambda; DynamoDB lưu state/history/idempotency/cost ledger; S3 lưu artifacts riêng tư |
| Chạy so sánh | SQS + Lambda worker; theo dõi run, chống submit trùng, kiểm quyền đọc kết quả |
| Retrieval | BM25; BGE-small-en-v1.5 384 chiều; hybrid bằng RRF; local cross-encoder reranker được đóng gói cho worker |
| Generation | DeepSeek API với đầu ra có cấu trúc; server kiểm và dựng citations từ nguồn đã lưu |
| Graph | Neo4j và typed source-backed graph experiments local; không có managed graph database đang chạy trên AWS |

Nguồn tiếng Anh được giữ làm bằng chứng; hỗ trợ hỏi/đáp tiếng Việt không đồng nghĩa BGE English là một encoder đa ngôn ngữ. Lịch sử và kết quả cũ được giữ theo phiên bản dữ liệu, không đổi nhãn thành corpus mới.

## 5. Những gì đang làm, chưa có trên bản AWS mới

- [x] Chuẩn bị candidate319 bằng cách bổ sung ba tài liệu chính thức theo khoảng trống đã phát hiện; tái sử dụng vectors cũ có kiểm tra.
- [x] Có triển khai thử nghiệm local cho tám phương pháp: BM25, Vector, Hybrid, Reranker, Graph 1-hop/2-hop, Multi-query (Q), Query-balanced (QB); cơ chế và độ đầy đủ bằng chứng vẫn đang được sửa/kiểm.
- [x] Làm tám trang giải thích phương pháp, trace từng bước và graph map từ dữ liệu có nguồn; các trang này chưa phát hành lên website AWS.
- [x] Chuẩn bị bộ 100 tình huống Việt–Anh và chạy chẩn đoán retrieval; kết quả có fallback/failure được giữ để sửa cơ chế, chưa công bố như benchmark hoàn chỉnh mới.
- [x] Xây và kiểm tra profile context 8/12 nguồn, output/citation protocol mới; bộ kiểm liên quan gần nhất có 266 tests đạt. Đây là focused tests local, không phải toàn bộ kiểm thử sản phẩm hoặc answer accuracy.
- [ ] Hoàn thiện task scope và cách tìm/chọn đủ nguồn cho Graph/Q/QB; xử lý zero expansion và lỗi minh bạch.
- [ ] Nối generation protocol mới với gateway/cache/ngân sách; kiểm câu trả lời, citations và abstention Việt–Anh trên candidate319.
- [ ] Tích hợp các phương pháp mới vào worker/API/UI, kiểm Linux/browser/quyền truy cập rồi review phát hành.
- [ ] Review độc lập nhãn và chất lượng câu trả lời trước khi dùng số đo như một kết luận tổng quát.

Graph coverage hiện chỉ là một tập con có nguồn chứng minh; không gọi đây là graph trích xuất đầy đủ toàn bộ 319 tài liệu. Các phiên bản Graph/Q/QB đang sửa không được trình bày thành capability live của website hiện tại. Bộ số đo pilot đã công bố vẫn có thể đọc ở [Evaluation](evaluation.md); đó là thí nghiệm khác, không thay cho kết quả 319.

## 6. Bằng chứng và giới hạn của báo cáo

| Mức bằng chứng | Quan sát | Điều không được suy ra |
|---|---|---|
| Kiểm mới ngày 28/10, 09:11 UTC+7 | Chrome DevTools mở trang AWS; nhận diện năm mục điều hướng và các tùy chọn official/snapshot, EN/VI, bốn live methods; HTTP readback khớp 13/13 assets của release 25/10; /v1/status không token trả 401 | Chưa kiểm lại phiên đăng nhập, gửi câu hỏi, email hoặc Live Compare mới trong lượt này |
| Receipts triển khai 25/10 | Owned workflow có 30 resources/8 JWT routes; hai canary qua API→queue→worker hoàn thành 5 method children, đối soát 40 source appearances | Không phải benchmark accuracy, kiểm tải lớn hoặc bài đo p95 |
| Xác nhận của người dùng 18–19/10 | Có answer tiếng Việt, nguồn/history; core account access sau email verification đã hoạt động | Không chứng minh mọi claim đúng, mọi thiết bị hoặc luồng reset password đã đạt |
| Thực nghiệm local đến 28/10 | Có pipeline, frozen receipts/replay và focused tests cho nhánh mới | Không đồng nghĩa code mới đã deploy hoặc các nhãn đã được người đánh giá độc lập duyệt |

Receipt public của lần đọc mới: [public-readback.json](../benchmarks/release-status-2026-10-28.json). Nó chứa HTTP status và SHA-256 của assets, không chứa token, tài khoản, nguồn riêng tư hoặc câu trả lời của người dùng. Các receipts vận hành chi tiết và nhật ký học giữ local; bảng này tóm tắt phạm vi kiểm, không giả rằng chúng đều đã được đưa lên GitHub.

**Phạm vi cập nhật GitHub lần này:** README, báo cáo tiến độ, receipt public và quy tắc cho phép file báo cáo. Sau push, CI phát hiện bốn test lịch sử phụ thuộc ngày thực tế khi bảng giá fixture hết hạn; bản sửa giới hạn ở hai module tests, cố định clock và giữ kiểm tra chặn giá chưa có hiệu lực/đã hết hạn. Không sửa runtime hoặc kéo dài thời hạn bảng giá để làm tests đạt. Không đẩy toàn bộ code local chưa review, corpus, weights, credentials, raw results hoặc nhật ký cá nhân. Không deploy lại AWS trong lượt báo cáo này.

## 7. Thứ tự demo để báo cáo với giảng viên — khoảng 5–7 phút

1. **Giới thiệu — 30 giây:** bài toán là hỏi tài liệu AWS và kiểm lại bằng chứng, kết hợp một đường live search với các snapshot có phiên bản để thí nghiệm.
2. **Tài khoản — 30 giây:** đăng nhập bằng tài khoản đã xác minh; không mở file credentials hoặc hiển thị OTP/mật khẩu. Không cần tạo tài khoản mới trong buổi báo cáo.
3. **Hỏi đáp — 1–2 phút:** Investigate → AWS official · live search → Tiếng Việt. Dùng câu đã lưu về RDS Multi-AZ/Read Replica để mở answer/source/history; nếu cần gửi câu mới thì gửi một lần và chờ.
4. **Giải thích bằng chứng — 1 phút:** chọn statement → đọc đoạn nguồn → mở link AWS. Nêu rõ nguồn hỗ trợ phần nào và điều kiện nào chưa xác định; Export brief nếu cần.
5. **So sánh retrieval — 1–2 phút:** Live Compare → câu tiếng Anh “Which policies must permit cross-account use of an AWS KMS key?” → dùng run đã hoàn tất, hoặc chạy BM25/Vector/Hybrid một lần nếu hạn mức còn cho phép. Mở source và saved trace. Live Compare chưa generation; không gọi overlap/rank là accuracy.
6. **Tiến độ tiếp theo — 30 giây:** chỉ rõ 319 và tám methods đang local; mục tiêu kế tiếp là đưa các phương pháp đã sửa vào cùng luồng có answer/citations, không tăng số tài liệu hoặc số đo để thay cho chất lượng.

Nếu email/provider/network gặp vấn đề, dùng tài khoản sẵn có và bản ghi/run đã hoàn tất để trình bày. Ghi rõ đang mở kết quả lưu; không diễn giải đó là lượt inference mới.

## 8. Đoạn báo cáo ngắn có thể sử dụng

“Đến hiện tại, em đã triển khai AWS Evidence Lab trên AWS, có đăng ký và xác minh email, hỏi đáp tài liệu AWS bằng tiếng Việt/Anh, trích dẫn nguồn, lịch sử riêng và xuất bản tóm tắt. Ngoài luồng hỏi đáp, em đã đưa lên web phần Live Compare để chạy và quan sát bốn phương pháp truy xuất trên 316 tài liệu chính thức. Em đang hoàn thiện GraphRAG, multi-query và query-balanced trên candidate319, đồng thời kiểm tra nguồn, các trường hợp thiếu bằng chứng và khả năng tái lập trước khi phát hành. Phần này vẫn được ghi là thực nghiệm local; chưa báo cáo như tính năng cloud hoàn chỉnh hoặc kết luận độ chính xác.”

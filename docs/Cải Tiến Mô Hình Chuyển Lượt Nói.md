# **Báo Cáo Nghiên Cứu Chuyên Sâu: Tối Ưu Hóa Dự Báo Chuyển Lượt (Turn-Taking Prediction) Cho Tiếng Việt Trên Thiết Bị Biên Thông Qua Học Giám Sát Yếu và Kiến Trúc Lai Ghép**

## **1\. Tóm Tắt Điều Hành**

Trong bối cảnh bùng nổ của các hệ thống Trí tuệ Nhân tạo Hội thoại (Conversational AI), khả năng tương tác tự nhiên, thời gian thực (real-time) giữa người và máy đang trở thành chuẩn mực mới. Tuy nhiên, rào cản lớn nhất ngăn cách giữa các trợ lý ảo hiện tại và một tác nhân giao tiếp thực thụ nằm ở cơ chế quản lý lượt lời (turn-taking management). Trong khi con người có khả năng chuyển giao lượt lời với độ trễ trung bình chỉ khoảng 200ms 5, các hệ thống hiện tại thường vận hành với độ trễ từ 700ms đến hơn 1,5 giây, tạo ra trải nghiệm giao tiếp rời rạc, thiếu tự nhiên và thường xuyên xảy ra tình trạng ngắt lời (barge-in) không mong muốn.7

Thách thức này càng trở nên phức tạp đối với tiếng Việt—một ngôn ngữ đơn lập, đa thanh điệu (tonal language) với hệ thống hư từ cuối câu phong phú. Sự giao thoa giữa thanh điệu từ vựng (lexical tone) và ngữ điệu câu (sentence intonation) tạo ra hiện tượng "nhiễu thanh điệu" (tonal interference), khiến các mô hình học máy truyền thống được huấn luyện trên dữ liệu tiếng Anh dễ dàng thất bại trong việc xác định Điểm Thích Hợp Chuyển Lượt (Transition Relevance Place \- TRP).

Báo cáo này cung cấp một phân tích toàn diện và đề xuất giải pháp kỹ thuật cụ thể để xây dựng mô hình dự báo chuyển lượt cho tiếng Việt, tập trung vào ba ràng buộc cốt lõi do người dùng đặt ra: (1) **Không gán nhãn thủ công** (sử dụng dữ liệu có sẵn và học giám sát yếu), (2) **Hiệu suất thời gian thực trên thiết bị biên** (Edge Devices), và (3) **Tối ưu hóa kiến trúc** thông qua việc thay thế GRU bằng TCN, sử dụng Focal Loss và cơ chế Soft-Gating.

Thông qua việc tổng hợp và phân tích dữ liệu từ hàng trăm tài liệu nghiên cứu và kho dữ liệu mở, báo cáo xác định **Kiến trúc Lai ghép Hạng nhẹ (Lightweight Hybrid Architecture)** là hướng đi tối ưu. Kiến trúc này kết hợp sức mạnh xử lý chuỗi thời gian song song của Mạng Tích chập Thời gian (Temporal Convolutional Networks \- TCN) với khả năng hiểu ngữ cảnh của các Mô hình Ngôn ngữ Nhỏ (SLM) đã được chưng cất, được điều phối bởi cơ chế cổng mềm (Soft-Gating) để xử lý sự bất định của tín hiệu đa phương thức. Đồng thời, báo cáo đề xuất quy trình **Học Giám sát Yếu (Weak Supervision)** sử dụng LLM để tự động gán nhãn cho các bộ dữ liệu hội thoại tự nhiên chưa được khai thác, giải quyết triệt để bài toán thiếu hụt dữ liệu huấn luyện.

## ---

**2\. Cơ Sở Lý Luận: Động Lực Học Hội Thoại và Thách Thức Ngôn Ngữ Tiếng Việt**

Để xây dựng một hệ thống AI có khả năng giao tiếp tự nhiên, trước hết cần giải mã cơ chế mà con người sử dụng để điều phối hội thoại, sau đó đối chiếu các cơ chế này với đặc thù ngôn ngữ học của tiếng Việt.

### **2.1 Mô hình Sacks, Schegloff và Jefferson (SSJ) và Nghịch lý 200ms**

Nền tảng của mọi nghiên cứu về luân phiên lượt lời hiện đại bắt nguồn từ công trình kinh điển của Sacks, Schegloff và Jefferson (1974). Họ đã thiết lập rằng hội thoại không phải là một chuỗi ngẫu nhiên các phát ngôn, mà là một cấu trúc có tổ chức chặt chẽ xoay quanh các **Đơn vị Cấu trúc Lượt (Turn-Constructional Units \- TCUs)**.7

Điểm mấu chốt trong mô hình SSJ là khái niệm **Điểm Thích Hợp Chuyển Lượt (Transition Relevance Place \- TRP)**. Đây là thời điểm mà một TCU hoàn tất về mặt cấu trúc, ngữ nghĩa và ngữ điệu, mở ra cơ hội cho việc chuyển giao quyền nói (floor transfer). Trong hội thoại tự nhiên, khoảng cách giữa các lượt lời (gap) trung bình chỉ khoảng **200ms**.5

**Nghịch lý nhận thức:** Các nghiên cứu tâm lý học ngôn ngữ chỉ ra rằng thời gian tối thiểu để não bộ con người lên kế hoạch và phát âm một từ đơn lẻ là khoảng **600ms**.5 Sự chênh lệch giữa con số 200ms của khoảng chuyển lượt và 600ms của thời gian chuẩn bị nói dẫn đến một kết luận quan trọng: Con người không chờ đối phương im lặng mới bắt đầu suy nghĩ. Chúng ta thực hiện cơ chế **Dự báo (Projection)**—dự đoán thời điểm kết thúc của câu nói ngay khi đối phương đang nói, dựa trên sự hội tụ của các tín hiệu cú pháp và ngữ điệu.7

Các hệ thống AI hiện tại thất bại vì chúng hoạt động theo cơ chế **Phản ứng (Reaction)** thay vì **Dự báo (Projection)**. Chúng chờ đợi một khoảng lặng (silence threshold) kéo dài từ 700ms đến 1500ms để xác nhận người dùng đã dứt lời. Điều này tạo ra một "khoảng chết" (dead air) trong hội thoại, làm giảm trải nghiệm người dùng và phá vỡ dòng chảy tự nhiên.10

### **2.2 Giả Thuyết Nhiễu Thanh Điệu (Tonal Interference) trong Tiếng Việt**

Một trong những thách thức lớn nhất khi áp dụng các mô hình dự báo chuyển lượt (như TurnGPT hay VAP) vào tiếng Việt là sự can nhiễu của thanh điệu. Hầu hết các mô hình SOTA hiện nay được tối ưu hóa cho các ngôn ngữ phi thanh điệu (non-tonal) như tiếng Anh, nơi ngữ điệu câu (intonation) hoạt động độc lập với từ vựng.11

Trong tiếng Anh, đường nét cao độ (F0 contour) ở cuối câu là chỉ dấu quan trọng nhất của TRP:

* **Falling Pitch (Xuống giọng):** Báo hiệu kết thúc câu trần thuật ![][image1] Nhả lượt (Yield).  
* **Rising Pitch (Lên giọng):** Báo hiệu câu hỏi hoặc sự chưa hoàn tất ![][image1] Giữ lượt (Hold).

Tuy nhiên, trong tiếng Việt, cao độ của từ cuối cùng bị ràng buộc bởi **thanh điệu từ vựng (lexical tone)**. Tiếng Việt có 6 thanh điệu (Ngang, Huyền, Sắc, Hỏi, Ngã, Nặng), và mỗi thanh điệu có một đường nét cao độ đặc trưng bắt buộc.12

**Bảng 1: Phân tích Tác động của Thanh điệu đến Tín hiệu Chuyển lượt**

| Loại Thanh Điệu | Đặc điểm Cao độ (F0) | Ý nghĩa trong Tiếng Anh | Nguy cơ Sai số Mô hình (False Prediction) |
| :---- | :---- | :---- | :---- |
| **Thanh Sắc / Ngã / Hỏi** | Đi lên mạnh hoặc gãy khúc (Rising/Broken) | Câu hỏi / Giữ lượt (Hold) | **False Negative (Bỏ lỡ TRP):** Mô hình nhầm tưởng người dùng chưa nói xong hoặc đang hỏi, dẫn đến việc chờ đợi vô ích. |
| **Thanh Huyền / Nặng** | Đi xuống thấp (Falling/Low) | Kết thúc câu (Yield) | **False Positive (Ngắt lời):** Mô hình nhầm tưởng người dùng đã dứt lời trong khi họ chỉ đang ngừng nghỉ nội tại (pause), dẫn đến barge-in. |
| **Thanh Ngang** | Phẳng, đều (Flat/Level) | Hesitation / Backchannel | **Backchannel Confusion:** Khó phân biệt giữa một câu trả lời ngắn và một tín hiệu phản hồi (backchannel) như "vâng", "ừ". |

Nghiên cứu của Hạ Kiều Phương và các cộng sự 14 chỉ ra rằng mặc dù ngữ điệu câu có tồn tại trong tiếng Việt, nó thường bị "lấn át" hoặc phải tương tác phức tạp với thanh điệu từ vựng. Ví dụ, một câu trần thuật kết thúc bằng từ "nhé" (thanh Sắc) sẽ luôn có cao độ đi lên, bất kể người nói có ý định nhường lượt hay không. Do đó, một mô hình chỉ dựa vào F0 thô (raw pitch) sẽ thất bại. Chúng ta cần một kiến trúc có khả năng **tách biệt (disentangle)** thanh điệu từ vựng khỏi ngữ điệu hội thoại.7

### **2.3 Vai Trò Của Hư Từ Cuối Câu (Sentence-Final Particles)**

Tiếng Việt sở hữu một hệ thống hư từ cuối câu (sentence-final particles \- SFPs) vô cùng phong phú, đóng vai trò như các "biển báo giao thông" rõ ràng cho luân phiên lượt lời.17

* **Nhóm Nhả lượt (Turn-Yielding):** *nhé, nhỉ, cơ, chứ, hả, à*. Khi xuất hiện ở cuối câu, các từ này gần như chắc chắn báo hiệu sự kết thúc của một lượt lời và mời gọi người nghe phản hồi.17  
* **Nhóm Giữ lượt/Kéo dài (Turn-Holding/Extension):** *mà, đã*. Các từ này có thể báo hiệu sự giải thích thêm hoặc yêu cầu chờ đợi (ví dụ: "Đợi mình chút đã").17  
* **Nhóm Phản hồi (Backchannels):** *ừ, vâng, dạ, thế à*. Đây là các tín hiệu cho thấy người nghe đang chú ý nhưng không muốn chiếm lượt lời.

Sự hiện diện của các hư từ này cung cấp một lợi thế đặc biệt cho tiếng Việt so với các ngôn ngữ khác: nếu mô hình có thể nhận diện chính xác các hư từ này (thông qua kênh văn bản hoặc âm thanh), độ chính xác của dự báo TRP sẽ tăng lên đáng kể. Điều này củng cố lập luận cho việc sử dụng **Mô hình Lai ghép (Hybrid Model)** kết hợp cả âm thanh và văn bản.

## ---

**3\. Chiến Lược Dữ Liệu: Khai Thác Tài Nguyên Sẵn Có & Học Giám Sát Yếu**

Một trong những rào cản lớn nhất được đặt ra trong yêu cầu nghiên cứu là **không gán nhãn thủ công**. Việc tạo ra một bộ dữ liệu hội thoại có gán nhãn TRP (Transition Relevance Place) thủ công đòi hỏi chi phí và thời gian khổng lồ. Do đó, báo cáo đề xuất chiến lược **Học Giám sát Yếu (Weak Supervision)** để tận dụng các nguồn dữ liệu tiếng Việt có sẵn.

### **3.1 Đánh Giá Các Bộ Dữ Liệu Tiếng Việt Hiện Có**

Chúng tôi đã tiến hành rà soát các bộ dữ liệu tiếng Việt công khai dựa trên tiêu chí về tính tự nhiên (spontaneity), sự hiện diện của hội thoại hai chiều (dyadic conversation) và khả năng khai thác cho bài toán turn-taking.

**Bảng 2: Đánh giá Tiềm năng Dữ liệu cho Turn-Taking**

| Tên Bộ Dữ Liệu | Loại Hình | Đặc điểm | Đánh giá Mức độ Phù hợp | Ghi chú |
| :---- | :---- | :---- | :---- | :---- |
| **VIVOS** 18 | Read Speech | 15h, 46 người nói. Đọc kịch bản có sẵn. | **Thấp** | Dữ liệu quá sạch, không có ngập ngừng, overlap hay backchannel. Không phản ánh đúng động lực hội thoại. |
| **FPT Open Speech (FOSD)** 20 | Read/Mixed | 30h. Chủ yếu là giọng đọc hoặc lệnh điều khiển. | **Trung bình \- Thấp** | Thiếu tính tương tác hai chiều tự nhiên. Hữu ích cho ASR nền tảng nhưng kém cho turn-taking. |
| **VLSP 2020 (100h)** 22 | **Spontaneous** & Read | \~80h hội thoại tự nhiên từ YouTube/MXH. | **Cao** | Chứa dữ liệu thực tế từ talkshow, phỏng vấn. Đây là nguồn dữ liệu quý giá nhất nhưng cần xử lý kỹ vì nhãn là transcript ASR, không phải nhãn hội thoại. |
| **ViVoice** 24 | Read Speech | Quy mô lớn (1000h+), giọng đọc audiobook. | **Thấp** | Tương tự VIVOS, thiếu tính chất tương tác thời gian thực. |
| **YouTube Podcasts (Vietcetera, etc.)** 7 | Conversational | Hội thoại tự nhiên, chất lượng âm thanh cao. | **Rất Cao** | Cần tự thu thập (crawl) nhưng phản ánh chính xác nhất bài toán turn-taking (có overlap, backchannel, laughter). |

**Kết luận về Dữ liệu:** Không có bộ dữ liệu nào có sẵn nhãn "Turn-End" hay "Backchannel" chuẩn. Tuy nhiên, bộ dữ liệu **VLSP 2020 (phần Spontaneous)** và các nguồn **Podcast tiếng Việt** là nguyên liệu thô lý tưởng để áp dụng quy trình tự động hóa nhãn.

### **3.2 Quy Trình Học Giám Sát Yếu (Weak Supervision Pipeline)**

Để giải quyết vấn đề không gán nhãn thủ công, chúng tôi đề xuất quy trình 4 bước sử dụng phương pháp **Data Programming** (như Snorkel) kết hợp với LLM để tạo ra các nhãn giả (pseudo-labels).25

#### **Bước 1: Tiền xử lý & Phân đoạn Người nói (Diarization)**

Sử dụng các mô hình Speaker Diarization đã được huấn luyện sẵn (như **pyannote.audio** hoặc **NVIDIA NeMo**) để phân tách luồng âm thanh thành các đoạn (segments) gắn với từng người nói.27

* *Mục tiêu:* Xác định chính xác thời điểm bắt đầu và kết thúc của từng lượt lời vật lý, phát hiện các đoạn chồng lấn (overlap).

#### **Bước 2: Nhận dạng Tiếng nói (ASR) với Time-alignment**

Sử dụng mô hình ASR mạnh mẽ như **Whisper-large-v3** hoặc **PhoWhisper** để chuyển đổi âm thanh thành văn bản.28

* *Yêu cầu quan trọng:* ASR phải xuất ra được **timestamps** (nhãn thời gian) cho từng từ (word-level timestamps). Điều này cho phép liên kết chính xác từ vựng (ví dụ: hư từ "nhé") với khung thời gian âm thanh tương ứng.

#### **Bước 3: Gán nhãn Tự động bằng LLM (LLM-as-a-Judge)**

Đây là bước đột phá thay thế con người. Sử dụng một LLM mạnh (như GPT-4o, Claude 3.5 Sonnet hoặc các mô hình tiếng Việt lớn như **PhoGPT-4B-Chat**) để phân tích transcript hội thoại.29

* **Prompting:** Cung cấp cho LLM đoạn hội thoại kèm theo ngữ cảnh và yêu cầu nó phân loại ranh giới của từng phát ngôn.  
  * *Input:* Transcript với thông tin người nói.  
  * *Task:* "Tại mỗi dấu ngắt câu, hãy xác định xem người nói đã kết thúc lượt (TRP), đang tạm dừng để suy nghĩ (Hold), hay chỉ đang phản hồi ngắn (Backchannel). Hãy chú ý đến các hư từ như 'nhé', 'nhỉ', 'à'."  
* **Output:** Nhãn ngữ nghĩa (Semantic Labels) cho từng mốc thời gian.

#### **Bước 4: Tổng hợp Nhãn (Label Aggregation)**

Kết hợp tín hiệu từ LLM với các **Hàm Gán Nhãn (Labeling Functions \- LFs)** dựa trên luật để tạo ra nhãn huấn luyện cuối cùng 25:

1. **LF\_Silence:** Nếu khoảng lặng sau từ \> 700ms ![][image1] Gán nhãn TRP.  
2. **LF\_Particle:** Nếu từ cuối cùng thuộc danh sách {*ạ, nhé, nhỉ, cơ, mà*} ![][image1] Gán nhãn TRP với độ tin cậy cao.17  
3. **LF\_LLM:** Sử dụng nhãn dự đoán từ Bước 3\.  
4. **LF\_Overlap:** Nếu có sự chồng lấn tiếng nói từ người kia ![][image1] Kiểm tra xem đó là Backchannel hay Interruption.

Mô hình Snorkel sẽ học trọng số của các LF này để tạo ra một bộ nhãn xác suất (probabilistic labels) có độ chính xác tiệm cận với gán nhãn thủ công mà không tốn sức người.

## ---

**4\. Đánh Giá Kỹ Thuật: TCN thay cho GRU**

Trong các kiến trúc xử lý chuỗi thời gian cho thiết bị biên, cuộc tranh luận giữa Mạng Nơ-ron Hồi quy (RNN/GRU) và Mạng Tích chập Thời gian (TCN) đang ngã ngũ với ưu thế nghiêng về TCN, đặc biệt cho các tác vụ thời gian thực.

### **4.1 Hạn Chế Của GRU (Gated Recurrent Unit)**

Mặc dù GRU nhẹ hơn LSTM (2 cổng so với 3 cổng) và giải quyết được phần nào vấn đề vanishing gradient, chúng vẫn tồn tại những nhược điểm cố hữu đối với bài toán này 32:

* **Xử lý Tuần tự (Sequential Processing):** GRU phải xử lý từng bước thời gian ![][image2] dựa trên trạng thái ẩn của ![][image3]. Điều này ngăn cản khả năng song song hóa (parallelism) trên các phần cứng hiện đại như GPU hoặc các bộ xử lý vector (NPU) trên chip di động, dẫn đến thông lượng (throughput) thấp.  
* **Bộ nhớ hạn chế:** Mặc dù lý thuyết là "vô hạn", thực tế GRU gặp khó khăn trong việc ghi nhớ các phụ thuộc rất dài nếu không có cơ chế Attention đi kèm, trong khi Attention lại làm tăng chi phí tính toán.  
* **Độ trễ suy luận:** Trên các thiết bị biên như Raspberry Pi, việc tính toán tuần tự của GRU thường chậm hơn so với các phép nhân ma trận được tối ưu hóa của CNN.

### **4.2 Ưu Điểm Vượt Trội Của TCN (Temporal Convolutional Networks)**

TCN, đặc biệt là biến thể **Causal Dilated TCN** (TCN nhân quả giãn nở), khắc phục được hầu hết các điểm yếu của GRU.34

* **Tính Nhân Quả (Causality):** TCN sử dụng *causal padding* để đảm bảo đầu ra tại thời điểm ![][image2] chỉ phụ thuộc vào các đầu vào từ ![][image2] trở về trước (![][image4]). Điều này đảm bảo tính hợp lệ cho các ứng dụng streaming thời gian thực, không có sự rò rỉ thông tin từ tương lai.  
* **Vùng Nhìn (Receptive Field) Linh Hoạt:** Thông qua cơ chế **Dilated Convolution** (tích chập giãn nở), vùng nhìn của TCN tăng theo hàm mũ (![][image5]) với độ sâu của mạng.  
  * *Công thức:* ![][image6].  
  * *Ý nghĩa:* Với một mạng nông (ít lớp), TCN vẫn có thể bao quát một cửa sổ lịch sử dài (ví dụ: 2-3 giây) để nắm bắt ngữ cảnh prosody (như xu hướng giảm cao độ) mà không tốn nhiều tham số như GRU.  
* **Hiệu Suất Tính Toán:** TCN thực chất là các phép nhân ma trận (Convolution), tận dụng cực tốt khả năng tính toán song song của các bộ tăng tốc phần cứng (NPU/DSP) trên thiết bị biên. Các nghiên cứu thực nghiệm trên Raspberry Pi cho thấy TCN có độ trễ suy luận thấp hơn và ổn định hơn so với LSTM/GRU cùng kích thước.36  
* **Ổn định Gradient:** TCN tránh được vấn đề bùng nổ/triệt tiêu gradient tốt hơn RNN do có đường truyền ngược (backpropagation) ngắn hơn và các kết nối thặng dư (residual connections).

**Kiến nghị:** Sử dụng kiến trúc **I-TCN (Independent Temporal-Aware Causal Network)** hoặc **Streaming TCN** với các lớp tích chập tách biệt theo chiều sâu (depthwise separable convolutions) để tối ưu hóa số lượng tham số cho thiết bị biên.37

## ---

**5\. Tối Ưu Hóa Hàm Mất Mát: Focal Loss**

Bài toán dự báo chuyển lượt có đặc điểm là sự **mất cân bằng dữ liệu (class imbalance)** cực đoan. Trong một cuộc hội thoại, phần lớn thời gian là "Speech" (tiếng nói liên tục) hoặc "Silence" (khoảng lặng ngắn). Các thời điểm chuyển lượt thực sự (TRP) là các sự kiện hiếm (rare events), chiếm tỷ lệ rất nhỏ trong tổng số khung hình (frames).39

Nếu sử dụng hàm mất mát **Cross-Entropy (CE)** tiêu chuẩn, mô hình sẽ có xu hướng dự đoán lớp đa số (không chuyển lượt) để đạt độ chính xác tổng thể cao, dẫn đến việc bỏ sót các TRP (False Negative) hoặc phản ứng chậm chạp.

### **5.1 Cơ Chế Của Focal Loss**

Focal Loss (![][image7]) được thiết kế để giải quyết vấn đề này bằng cách giảm trọng số của các mẫu "dễ" (đã được phân loại đúng với độ tin cậy cao) và tập trung vào các mẫu "khó" (các điểm chuyển tiếp mơ hồ).41

Công thức toán học:

![][image8]  
Trong đó:

* ![][image9]: Xác suất mô hình dự đoán đúng lớp (ground truth).  
* ![][image10]: **Hệ số điều chỉnh (Modulating factor)**.  
  * Khi ![][image11] (mẫu dễ, ví dụ: giữa câu nói rõ ràng), hệ số này tiến về 0 ![][image1] Loss đóng góp không đáng kể.  
  * Khi ![][image9] thấp (mẫu khó, ví dụ: cuối câu có thanh Sắc gây nhiễu), hệ số này lớn ![][image1] Loss đóng góp lớn, buộc mô hình phải học kỹ.  
* ![][image12] (Gamma): Tham số tập trung (thường chọn ![][image13]).  
* ![][image14] (Alpha): Tham số cân bằng lớp.

### **5.2 Lợi Ích Đối Với Tiếng Việt**

Trong tiếng Việt, Focal Loss đóng vai trò sống còn để xử lý **nhiễu thanh điệu**.

* Các trường hợp "dễ" (câu kết thúc bằng thanh Huyền xuống giọng) sẽ bị Focal Loss giảm trọng số.  
* Các trường hợp "khó" (câu kết thúc bằng thanh Sắc lên giọng nhưng lại là điểm cuối câu) sẽ tạo ra loss lớn. Focal Loss ép mô hình không được "lười biếng" dựa vào quy tắc "lên giọng \= chưa hết", mà phải tìm kiếm các đặc trưng tinh tế hơn (như độ dài âm tiết, năng lượng, hoặc hư từ đi kèm) để phân loại đúng.7

## ---

**6\. Cơ Chế Hợp Nhất: Soft-Gating Đa Phương Thức**

Để đạt độ chính xác cao nhất, hệ thống cần kết hợp cả tín hiệu âm thanh (Acoustic) và văn bản (Linguistic). Tuy nhiên, độ tin cậy của hai nguồn này thay đổi liên tục theo thời gian thực (ví dụ: môi trường ồn làm âm thanh kém tin cậy, hoặc ASR bị trễ làm văn bản kém tin cậy).

### **6.1 Tại Sao Chọn Soft-Gating?**

* **Hard-Gating (Cổng Cứng):** Sử dụng quy tắc logic (IF-THEN). Ví dụ: "Nếu xác suất âm thanh \> 0.8 thì dùng âm thanh". Phương pháp này cứng nhắc, dễ bị "giật" (jitter) ở các điểm ngưỡng và không tận dụng được sự bổ trợ thông tin.43  
* **Soft-Gating (Cổng Mềm):** Sử dụng một mạng nơ-ron nhỏ để học cách gán trọng số (![][image15]) liên tục cho từng phương thức dựa trên ngữ cảnh hiện tại. Nó cho phép mô hình "trượt" giữa việc tin vào âm thanh hay văn bản một cách mượt mà.44

### **6.2 Mô Hình Gated Multimodal Unit (GMU)**

Báo cáo đề xuất sử dụng kiến trúc **GMU** để thực hiện Soft-Gating. Giả sử ![][image16] là vector đặc trưng từ TCN (âm thanh) và ![][image17] là vector từ mô hình ngôn ngữ (văn bản).

Cơ chế hoạt động như sau:

1. **Tính toán Cổng (![][image15]):** Một mạng nơ-ron học cách quyết định xem tại thời điểm này, nguồn tin nào đáng tin cậy hơn.  
   ![][image18]  
   (![][image19] là hàm Sigmoid, đầu ra từ 0 đến 1).  
2. **Hợp nhất Đặc trưng (![][image20]):**  
   ![][image21]

**Ứng dụng cho Backchannel Tiếng Việt:**

* Khi người dùng nói "Ừ" (Backchannel):  
  * Mô hình văn bản (Linguistic) có thể thấy từ "Ừ" là một câu hoàn chỉnh về ngữ pháp ![][image1] Dự báo: Turn End.  
  * Mô hình âm thanh (Acoustic TCN) thấy cao độ phẳng, cường độ thấp, thời gian cực ngắn ![][image1] Dự báo: Backchannel (Tiếp tục nghe).  
  * **Cổng Soft-Gating:** Sẽ học được rằng trong tình huống này (ngắn, thấp), đặc trưng âm thanh đáng tin hơn. Nó điều chỉnh ![][image15] để ưu tiên ![][image16], giúp hệ thống không ngắt lời người dùng vô duyên. Đây là chìa khóa để giải quyết vấn đề nhập nhằng ngữ nghĩa/ngữ âm trong tiếng Việt.46

## ---

**7\. Kiến Trúc Hệ Thống Đề Xuất & Triển Khai**

Dựa trên các phân tích trên, chúng tôi đề xuất kiến trúc hệ thống tổng thể như sau:

### **7.1 Thành Phần Hệ Thống**

1. **Acoustic Stream (Luồng Âm thanh):**  
   * **Input:** Log-Mel Spectrogram (trích xuất từ cửa sổ trượt 20ms).  
   * **Backbone:** **Causal Dilated TCN** (4 lớp, kernel size 3, dilation 1). Đảm bảo Receptive Field khoảng 2-3 giây để bắt trọn ngữ điệu câu.  
   * **Output:** Vector ![][image16].  
2. **Linguistic Stream (Luồng Ngôn ngữ):**  
   * **Input:** Token văn bản từ Streaming ASR (Whisper-Tiny hoặc PhoWhisper-Small).  
   * **Backbone:** **DistilPhoBERT** (phiên bản chưng cất của PhoBERT) hoặc **TinyBERT** tiếng Việt. Mô hình này cần được lượng tử hóa (Quantization) xuống INT8 để chạy nhanh trên CPU.48  
   * **Output:** Vector ![][image17].  
3. **Fusion Layer (Tầng Hợp nhất):**  
   * Sử dụng **GMU Soft-Gating** để kết hợp ![][image16] và ![][image17].  
   * Hàm kích hoạt đầu ra: Softmax cho 3 lớp: Turn-Keep, Turn-Yield, Backchannel.  
4. **Training:**  
   * Sử dụng **Focal Loss** để huấn luyện toàn mạng.  
   * Dữ liệu: Dataset tự tạo từ quy trình Weak Supervision (Phần 3.2).

### **7.2 Đánh Giá Hiệu Năng trên Thiết Bị Biên (Raspberry Pi 4/5)**

* **Mô hình TCN:** Thời gian suy luận dự kiến \< 5ms/frame (so với 10-15ms của GRU).  
* **Mô hình DistilPhoBERT (INT8):** Thời gian suy luận dự kiến \< 40ms.50  
* **Tổng độ trễ (Latency):** \< 100ms cho việc xử lý mô hình dự báo. Cộng thêm độ trễ ASR (thường là bottleneck chính), hệ thống có thể đạt phản hồi trong khoảng 300-500ms, tiến gần đến ngưỡng tự nhiên của con người hơn nhiều so với các hệ thống VAD truyền thống (1000ms+).  
* **Gap Deviation:** Sử dụng metric "Gap Deviation" để đo độ lệch chuẩn giữa khoảng lặng dự báo và khoảng lặng thực tế trong tập test, nhằm đảm bảo mô hình không chỉ đúng mà còn "đúng lúc".51

## ---

**8\. Kết Luận**

Việc xây dựng một hệ thống dự báo chuyển lượt cho tiếng Việt mà không cần gán nhãn thủ công là hoàn toàn khả thi thông qua sự kết hợp của các kỹ thuật tiên tiến: **Học giám sát yếu với LLM**, **Kiến trúc TCN nhân quả**, và **Cơ chế Soft-Gating**.

Giải pháp này không chỉ giải quyết được các thách thức đặc thù của tiếng Việt (nhiễu thanh điệu, hư từ) mà còn đáp ứng được yêu cầu khắt khe về tài nguyên của thiết bị biên. Thay vì phụ thuộc vào các mô hình "khổng lồ" trên đám mây, hướng đi "nhỏ mà thông minh" (Small & Smart) sử dụng kiến trúc lai ghép sẽ là chìa khóa để đưa AI hội thoại tiếng Việt bước sang một kỷ nguyên mới về độ tự nhiên và linh hoạt.

**Khuyến nghị triển khai:** Bắt đầu bằng việc thu thập dữ liệu từ tập VLSP 2020 Spontaneous và các kênh Podcast, chạy quy trình Weak Supervision để tạo dữ liệu huấn luyện (Silver Dataset). Sau đó, tập trung huấn luyện nhánh Acoustic TCN trước vì nó nhẹ và chạy nhanh nhất, trước khi tích hợp nhánh Linguistic phức tạp hơn.

#### **Works cited**

1. Gemini Live API models high Latency \- Google AI Developers Forum, accessed January 12, 2026, [https://discuss.ai.google.dev/t/gemini-live-api-models-high-latency/108989](https://discuss.ai.google.dev/t/gemini-live-api-models-high-latency/108989)  
2. How I Cut Voice Chat Latency by 23% Using Parallel LLM API Calls : r/ChatGPTPro \- Reddit, accessed January 12, 2026, [https://www.reddit.com/r/ChatGPTPro/comments/1l72sxb/how\_i\_cut\_voice\_chat\_latency\_by\_23\_using\_parallel/](https://www.reddit.com/r/ChatGPTPro/comments/1l72sxb/how_i_cut_voice_chat_latency_by_23_using_parallel/)  
3. WhisperFlow: speech foundation models in real time \- arXiv, accessed January 12, 2026, [https://arxiv.org/pdf/2412.11272](https://arxiv.org/pdf/2412.11272)  
4. An Evaluation of LLMs Inference on Popular Single-board Computers \- arXiv, accessed January 12, 2026, [https://arxiv.org/html/2511.07425v1](https://arxiv.org/html/2511.07425v1)  
5. Timing in turn-taking and its implications for processing models of language \- Frontiers, accessed January 12, 2026, [https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00731/full](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00731/full)  
6. The intersection of turn-taking and repair: the timing of other-initiations of repair in conversation \- PubMed Central, accessed January 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4357221/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4357221/)  
7. Tìm Khoảng Trống Nghiên Cứu Paper.docx  
8. The intersection of turn-taking and repair: the timing of other-initiations of repair in conversation \- Frontiers, accessed January 12, 2026, [https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00250/full](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00250/full)  
9. Long gaps between turns are awkward for strangers but not for friends | Philosophical Transactions of the Royal Society B, accessed January 12, 2026, [https://royalsocietypublishing.org/rstb/article/378/1875/20210471/109157/Long-gaps-between-turns-are-awkward-for-strangers](https://royalsocietypublishing.org/rstb/article/378/1875/20210471/109157/Long-gaps-between-turns-are-awkward-for-strangers)  
10. TURN-TAKING AND BACKCHANNEL PREDICTION WITH ACOUSTIC AND LARGE LANGUAGE MODEL FUSION \- Amazon Science, accessed January 12, 2026, [https://assets.amazon.science/95/b2/0cd8a6ce484497c31a7cf932ae3c/turn-taking-and-backchannel-prediction-with-acoustic-and-large-language-model-fusion.pdf](https://assets.amazon.science/95/b2/0cd8a6ce484497c31a7cf932ae3c/turn-taking-and-backchannel-prediction-with-acoustic-and-large-language-model-fusion.pdf)  
11. 3.12 Tone and intonation – Essentials of Linguistics, 2nd edition \- eCampusOntario Pressbooks, accessed January 12, 2026, [https://ecampusontario.pressbooks.pub/essentialsoflinguistics2/chapter/3-12-tone-and-intonation/](https://ecampusontario.pressbooks.pub/essentialsoflinguistics2/chapter/3-12-tone-and-intonation/)  
12. “Do You Hear What I Hear?”: On the Elusive Nature of Vietnamese Tones, accessed January 12, 2026, [https://libjournals.unca.edu/ncur/wp-content/uploads/2021/02/3001-Nguyen-Mai-Chi-FINAL.pdf](https://libjournals.unca.edu/ncur/wp-content/uploads/2021/02/3001-Nguyen-Mai-Chi-FINAL.pdf)  
13. Mastering the Six Tones of Vietnamese – The Best Guide to Pronunciation, accessed January 12, 2026, [https://www.catalystforchangevietnam.com/post/mastering-the-six-tones-of-vietnamese-the-best-guide-to-pronunciation](https://www.catalystforchangevietnam.com/post/mastering-the-six-tones-of-vietnamese-the-best-guide-to-pronunciation)  
14. Modelling the Interaction of Intonation and Lexical Tone in Vietnamese, accessed January 12, 2026, [https://ifl.phil-fak.uni-koeln.de/sites/linguistik/Phonetik/pdf-publications/2010/Ha\_Grice-Modelling\_the\_Interaction\_of\_Intonation\_and\_Lexical\_Tone\_in\_Vietnamese\_2010.pdf](https://ifl.phil-fak.uni-koeln.de/sites/linguistik/Phonetik/pdf-publications/2010/Ha_Grice-Modelling_the_Interaction_of_Intonation_and_Lexical_Tone_in_Vietnamese_2010.pdf)  
15. Modelling the Interaction of Intonation and Lexical Tone in Vietnamese \- ResearchGate, accessed January 12, 2026, [https://www.researchgate.net/publication/266443897\_Modelling\_the\_Interaction\_of\_Intonation\_and\_Lexical\_Tone\_in\_Vietnamese](https://www.researchgate.net/publication/266443897_Modelling_the_Interaction_of_Intonation_and_Lexical_Tone_in_Vietnamese)  
16. ANOTHER LOOK AT VIETNAMESE INTONATION \- International Phonetic Association, accessed January 12, 2026, [https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS1999/papers/p14\_2399.pdf](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS1999/papers/p14_2399.pdf)  
17. Teaching Final Particles in Vietnamese \- researchmap, accessed January 12, 2026, [https://researchmap.jp/TrongGiang-2311/published\_papers/26744448/attachment\_file.pdf](https://researchmap.jp/TrongGiang-2311/published_papers/26744448/attachment_file.pdf)  
18. VIVOS: Vietnamese Speech Corpus for ASR \- Kaggle, accessed January 12, 2026, [https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr)  
19. VINH123/tiengviet · Datasets at Hugging Face, accessed January 12, 2026, [https://huggingface.co/datasets/VINH123/tiengviet](https://huggingface.co/datasets/VINH123/tiengviet)  
20. doof-ferb/fpt\_fosd · Datasets at Hugging Face, accessed January 12, 2026, [https://huggingface.co/datasets/doof-ferb/fpt\_fosd](https://huggingface.co/datasets/doof-ferb/fpt_fosd)  
21. FPT Open Speech Dataset (FOSD) \- Vietnamese \- Mendeley Data, accessed January 12, 2026, [https://data.mendeley.com/datasets/k9sxg2twv4/4](https://data.mendeley.com/datasets/k9sxg2twv4/4)  
22. VinBigdata shares 100-hour data for the community, accessed January 12, 2026, [https://vinbigdata.com/en/news/vinbigdata-shares-100-hour-data-for-the-community](https://vinbigdata.com/en/news/vinbigdata-shares-100-hour-data-for-the-community)  
23. doof-ferb/vlsp2020\_vinai\_100h · Datasets at Hugging Face, accessed January 12, 2026, [https://huggingface.co/datasets/doof-ferb/vlsp2020\_vinai\_100h](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h)  
24. Zero-Shot Text-to-Speech for Vietnamese \- arXiv, accessed January 12, 2026, [https://arxiv.org/pdf/2506.01322](https://arxiv.org/pdf/2506.01322)  
25. Essential Guide to Weak Supervision | Snorkel AI, accessed January 12, 2026, [https://snorkel.ai/data-centric-ai/weak-supervision/](https://snorkel.ai/data-centric-ai/weak-supervision/)  
26. Language Models in the Loop: Incorporating Prompting into Weak Supervision | Request PDF \- ResearchGate, accessed January 12, 2026, [https://www.researchgate.net/publication/379684928\_Language\_Models\_in\_the\_Loop\_Incorporating\_Prompting\_into\_Weak\_Supervision](https://www.researchgate.net/publication/379684928_Language_Models_in_the_Loop_Incorporating_Prompting_into_Weak_Supervision)  
27. The 2025 VLSP Task on Vietnamese Voice Conversion: Overview and Preliminary Results \- ACL Anthology, accessed January 12, 2026, [https://aclanthology.org/2025.vlsp-1.13.pdf](https://aclanthology.org/2025.vlsp-1.13.pdf)  
28. Whisper based Cross-Lingual Phoneme Recognition between Vietnamese and English, accessed January 12, 2026, [https://arxiv.org/html/2508.19270v1](https://arxiv.org/html/2508.19270v1)  
29. Automatic Labelling with Open-source LLMs using Dynamic Label Schema Integration, accessed January 12, 2026, [https://arxiv.org/html/2501.12332v1](https://arxiv.org/html/2501.12332v1)  
30. Using LLMs for Automated Data Labeling \- Damco Solutions, accessed January 12, 2026, [https://www.damcogroup.com/blogs/automated-data-labeling-with-llms](https://www.damcogroup.com/blogs/automated-data-labeling-with-llms)  
31. Applying General Turn-taking Models to Conversational Human-Robot Interaction \- arXiv, accessed January 12, 2026, [https://arxiv.org/html/2501.08946v1](https://arxiv.org/html/2501.08946v1)  
32. TCN-GRU Based on Attention Mechanism for Solar Irradiance Prediction \- MDPI, accessed January 12, 2026, [https://www.mdpi.com/1996-1073/17/22/5767](https://www.mdpi.com/1996-1073/17/22/5767)  
33. TCN\_EDGE | PDF | Deep Learning | Algorithms \- Scribd, accessed January 12, 2026, [https://www.scribd.com/document/945264939/TCN-EDGE](https://www.scribd.com/document/945264939/TCN-EDGE)  
34. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling \- arXiv, accessed January 12, 2026, [https://arxiv.org/pdf/1803.01271](https://arxiv.org/pdf/1803.01271)  
35. Dilated Causal Convolutional Networks \- Emergent Mind, accessed January 12, 2026, [https://www.emergentmind.com/topics/dilated-causal-convolutional-networks](https://www.emergentmind.com/topics/dilated-causal-convolutional-networks)  
36. Benchmarking deep learning models for ET 0 forecasting on edge devices \- IWA Publishing, accessed January 12, 2026, [https://iwaponline.com/hr/article/56/12/1269/110480/Benchmarking-deep-learning-models-for-ET0](https://iwaponline.com/hr/article/56/12/1269/110480/Benchmarking-deep-learning-models-for-ET0)  
37. A Lightweight Forward-backward Independent Temporal-aware Causal Network for Speech Emotion Recognition \- IEEE Xplore, accessed January 12, 2026, [https://ieeexplore.ieee.org/iel8/6287639/6514899/10993381.pdf](https://ieeexplore.ieee.org/iel8/6287639/6514899/10993381.pdf)  
38. AudioRepInceptionNeXt: A lightweight single-stream architecture for efficient audio recognition \- arXiv, accessed January 12, 2026, [https://arxiv.org/html/2404.13551v1](https://arxiv.org/html/2404.13551v1)  
39. 6.4. Classification on imbalanced labels with focal loss \- skscope, accessed January 12, 2026, [https://skscope.readthedocs.io/en/0.1.7/gallery/Miscellaneous/focal-loss-with-imbalanced-data.html](https://skscope.readthedocs.io/en/0.1.7/gallery/Miscellaneous/focal-loss-with-imbalanced-data.html)  
40. An Asymmetric Contrastive Loss for Handling Imbalanced Datasets \- PMC, accessed January 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9497504/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9497504/)  
41. How to Fix Imbalanced Classes with Focal Loss | by Natthawat Phongchit | Medium, accessed January 12, 2026, [https://medium.com/@natthawatphongchit/how-to-fix-imbalanced-classes-with-focal-loss-559de3ef94a3](https://medium.com/@natthawatphongchit/how-to-fix-imbalanced-classes-with-focal-loss-559de3ef94a3)  
42. Adaptive Focal Loss for Keypoint-Based Deep Learning Detectors Addressing Class Imbalance \- IEEE Xplore, accessed January 12, 2026, [https://ieeexplore.ieee.org/iel8/6287639/10820123/10872927.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/10872927.pdf)  
43. Hard vs Soft Gating and the Benefits of Soft Gating | by Manpreet Wadan | AIAutomation, accessed January 12, 2026, [https://medium.com/aiautomation/hard-vs-soft-gating-and-the-benefits-of-soft-gating-7fa09d099a90](https://medium.com/aiautomation/hard-vs-soft-gating-and-the-benefits-of-soft-gating-7fa09d099a90)  
44. gated multimodal units for information fu \- arXiv, accessed January 12, 2026, [https://arxiv.org/pdf/1702.01992](https://arxiv.org/pdf/1702.01992)  
45. Gated Fusion Mechanisms \- Emergent Mind, accessed January 12, 2026, [https://www.emergentmind.com/topics/gated-fusion-mechanism](https://www.emergentmind.com/topics/gated-fusion-mechanism)  
46. Gated multimodal networks | Request PDF \- ResearchGate, accessed January 12, 2026, [https://www.researchgate.net/publication/338610499\_Gated\_multimodal\_networks](https://www.researchgate.net/publication/338610499_Gated_multimodal_networks)  
47. (PDF) A Practical Multimodal Fusion System With Uncertainty Modeling for Robust Visual and Affective Applications \- ResearchGate, accessed January 12, 2026, [https://www.researchgate.net/publication/394559337\_A\_Practical\_Multimodal\_Fusion\_System\_with\_Uncertainty\_Modeling\_for\_Robust\_Visual\_and\_Affective\_Applications](https://www.researchgate.net/publication/394559337_A_Practical_Multimodal_Fusion_System_with_Uncertainty_Modeling_for_Robust_Visual_and_Affective_Applications)  
48. PhoBERT: Pre-trained language models for Vietnamese | Semantic Scholar, accessed January 12, 2026, [https://pdfs.semanticscholar.org/74fc/832dd6c77253595cf3c1c852045c8da93c13.pdf](https://pdfs.semanticscholar.org/74fc/832dd6c77253595cf3c1c852045c8da93c13.pdf)  
49. Efficient Transformer Knowledge Distillation: A Performance Review \- arXiv, accessed January 12, 2026, [https://arxiv.org/html/2311.13657](https://arxiv.org/html/2311.13657)  
50. Optimizing BERT for Android \- Medium, accessed January 12, 2026, [https://medium.com/@ktlint/optimizing-bert-for-android-ef01dbf45cb0](https://medium.com/@ktlint/optimizing-bert-for-android-ef01dbf45cb0)  
51. Automated extraction of speech and turn-taking parameters in autism allows for diagnostic classification using a multivariable prediction model \- Frontiers, accessed January 12, 2026, [https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2023.1257569/full](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2023.1257569/full)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAfUlEQVR4XmNgGAWjYHACWSDuBmIOdAlyQTkUUwWIAfF+IDZDlyAXgAw6AsQq6BI8QCxJBg4G4kdAzMmABCqggqTiZ0D8H4jjGSgE3EC8EIj70CVIBa5AvJoBzXvkABYGiIs80CXIAdJAvBmIRdAlyAGsQCwExIzoEqNggAEAkekYp+CjMnEAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAbCAYAAACwRpUzAAAAkElEQVR4XmNgGAJABV0ABhSBeCG6IAx4AnE5uiAMtAKxC7ogCHAA8VYglkYW5AFiSSCOAOJ/ULYYEDMjKwLZ9R9ZAAZYgHgNED9HlwABcSC+C8QH0MTBwAaIfwPxJHQJEChigNgXxACxAuQ4YZAEzL63QKwJxMZAvBiIOcHagMAXiD8C8QYGSChhAJh/RxwAAAqIErJPedn0AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAZCAYAAABD2GxlAAAA9ElEQVR4Xu2VOwrCQBRFR2wUxQ+CImrjFrQRxCXYirWFjdZZhqALEAuxsHYBVhZuQt2DlaDeYTIkeUkmaXyNc+BAmPtCbiYDEcJisfgpwQ5d9CMHRjBPAwam8A53NNC04M1VXnNRgXXowI8wFBzCFzzBHMk4SCyoBxY0YCK24MwNqEX/EAOxBTVvuIcZGjCRWFCGc7oYgXyBGmymNO2XMBYswCfs0SAC+cAjfKR0qW5LxFiwC6+wSgNGjAXHcCu889eGWS9mwVhwDQdCFZzATTBmYSVUQXl8Que2Dy/wDA+wHIx/it45amgnG0L9diwWi+Wf+AKpaz6I4IfNvwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFcAAAAZCAYAAABEmrJwAAABwUlEQVR4Xu2XvUoDQRSFr6igaKOIIgqpFKwsFAvxBSxsbC3zAoKgL2IhKIi1diI2FoJ2gjYWNkKwVCwUBEH8OYfJknhIyGQ37E5gPviauRwyuZvMnTWLRCKRSCQdo7oQEiW4DHu04EmWbBaGYBm+aSEUFuAHPIMDUvOB+bTZLIzDXngEf6UWDBvmNrejBU+YT5vtBEE39wB+wRUteMJ82mwnCLK5PKu4KdWHZtkifsFBNjchS1OSI6VIgm0uJ3zaI4HZQ3P5VnDwcABNetoOwTZ3BFbglKz7wOyNuXwrZuEtfPK0z8W8CLa58/DE2vsyCcy+m8sXSbDN1WsUXwZ80Sscs7zz5k2wzeU1ig3ph3twq66WbLpZw5jlywfrms0LvqGdmtunvsQMw0tztcq/So01q910GsHj7MVcnf1Q+PkX5uqPUrNFeA7v4La5wZOwDj/NbaARzD6by2s2D5KmqMl+OXD5wL/ha3VNKcF7+KCFKoPmhjb7wH40YhP+wF0tEE5nPuVmrOpCHfy1tDvd84bzZF8XQ4BPbk4Xu4wJeKyLRcO/zLUudhlL8ApOa6FoeFjP6GKXMWYBNjYSiUQineMPfsNo3H2UaZcAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABEklEQVR4XmNgoCNgBOIAdEFiATcQ70IXpAqQAWJ7IJYEYmYkcZBzU4F4OxBvQRKHA1MgfgbEr4D4PxB/BOJwqJwiEGcBsQkQX4WKwYEhEN9C4lsB8Xsg/gvl6wKxJhCXA/EamCIQADlpChCfQRYEghwGiAvEoXweID4AxNEwBciCIIUgNgyAvPENiF2gfEsgvg7ESkDMAlMEAiDbeZEFGCCa/gGxMZQfBsSbgXgCXAUOAPMKyDWsSOL8aHyswByIvzKgBiJRQJ4BoukKlE00ADlrDxAfAmIJNDmCYBYQr2aAGAID0khsrAAUQGVALMcASZowrMcAiRq8IJgBkppAoYuOBZHUYQDkRIINc8BVjoIBBACjtTCYw+7cogAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAZCAYAAAAiwE4nAAABWklEQVR4Xu2TsSvFURTHj1AML5RNegaLiZJBGRSzyWYxMTA/E5uU0SgyyWIw8A9YDRayvozKQLEofL/dc7zzu+93b8+k9PvUp37nnHvu/b1z30+k4r8xBQ87dFB7jD14rDXvBRx16wrU4TLcgV/wQGOzAW+1VtMeY17CmmsJ9TWNF2Bva1k5POgK9sUFZStOKBPwGX7GhRxD8EbSm5JUbUnCr2tG+SyT8BXORfld97ztnj2cDA88jws5ViQ0jbhcP1x0cRk2GfamJtDGMLyX0PSoPmmcuk/DXpT93KcjpuEb/HC5Lnjq4hQnEg7kWvbEcALdcXJdWm9p9MAzF6ewyXCPMi7hgE9wY142m/i2Pj/r4hSczDuciQvKZpzgn6Qp4UDex2/J3d+4hG+0wL6Epjs4Vixl4X3xU2LvqsvX4QZ80doP9t0x6eW9cZw5jqS9r8wHa6ioqPh7vgHiMleDJKP5ywAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAABCklEQVR4XmNgGAWjABWIATEzEl8ciDmQ+DgBJxCvAWJfIP4PxL+AuB4q5wHEf4E4A8rHCiwZIAZEM0AMmA/ErFA5YyD+CsR7oHyswAqI/RggGkEGgNgwYAPEv4H4FpIYTgCziRvKZ2GAuAxk6CSoGMiVV4FYBMpHASCFrUh8RSB+AsQ/GSDeBAGQQcsZIIajAJAAyABQQMIAKOBAYhOAmBGIjzFAAvgREG9HUgcGIL+eAOIDQHwYiDcCcTEDIjBBAK/zy6EYZJMwAyQNoAOQ85cyQNSgAFBi2QrELugSaOA0EKdD2SrIEtJA/ABK4wMPGSBeBbmwAVkiBAlrIkugAVAyB2lGTu6jAAgAH8EtJg89L0sAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAZCAYAAAB9/QMrAAAC00lEQVR4Xu2YzatNURjGXyHkM0SKLmKgxMBHkYEBIWVCIR8DExMfA3UVxZ3IUImIdJNkwMRAKOXEgD+AlDIgZagUkcTz3Hete9Z+71r77u04d9+t/aune8+71j577We9613rbJGGhm4y1ul/YTo03gY7gV92GRpjG2rMduiJDf4tNOgKdMM2BEyDFthgDdgP9dggmA89g35Dq11sEnRmsIdhJ/QOWmIbwERoL/Qeumna6gAT4Jr765kFPYXOQbugB6IGbYIOBf0GOQJ9hpbbBjBDtEadFHW8jiaRR040IgbNOQ8dtw2elqiTzJgUdTfpMPQDWmcbHKugNxJPlAG+Qqds0FC1SXMku+vOlfxJtayBvok+RwyadBUaZxs8fPhtNmio0iQWVdbDV9BS6IRoweVDn5ZiW/w80e+4J3EjNkAHbTDkE7TYBg1VmcTlwQfbJ3r/fmmbwtnnKiiyxTPrWFJoFA2z3JJhPEhdGFLGpCmi31dUeeey9dAOUXN4f/7v4ez/hN4GsTw49tSzXpd0UR9Y56kLQ8qYdBT6UEI0dThey9CMZzHmmFru80bRfqxXMfJMWmQDltSFIWVM6gZ+WU12n1lXuAw5posuxjNPapdmtt6WYs8a5Qu00gYNVZvEe9MED2f+o7S3dV9z+oI+IczWluiBOZVpuXAALIwpOIALov3umraRYLbovS9Ju37x8Mti6zOLh0EawJ8aMbhMuVx91pWGM5K62GeQ1UhmFAv0S9FMeA7dFz0GhFt/3lIjLPi/oK22oSh3RAseZ2w0womimEX8zRVbLjQo70DMJOCJOnZtITZD36UDl7uIrzVcTnmwIPs+K8IGBw3qs8Gy9IieaHfbhorg7nUMeii6vPuhPZkeWV5AZ0Xfh/FFW8hC0bcc/4S1Uvxg1m0mQFtEX2N4Lcv0yML6NFOGHkxp2ONIvCM4kKk2WGN6oQM22NDQMOr4AxjxlViDOVafAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAZCAYAAACVfbYAAAABsUlEQVR4Xu2WvytGURjHH6EIEfLbwKYkxSgkBoMUNgblDzAqC4tZKQZZGBiMyGKz+AMohcXAZDArfL+ee+s4vffec3mv3lfnU5/ezvOc+97zdH5dEY/H4/kZ9bDJDv4HluArXLETxUyd6Gztww8p0OIG4aodTEFicQ2wJKadJV3wHDbaCUdiixuCT6Iv6IR78BrewX6jX1aUwR04YycciSyOhR3ADdEOi0aOnRmbNGJZMit66qUlsrgJ0cFfwBfYY+QiH3KAm701pW3wFk7DUnEncZzP8BLWBG3+ss2HpoLYKLyBzUE7igq4DR9/IN93D8fEncTi3uGW0eYMcibN2eTSPRMdfBa0wGVJN2sksTgmzQ09H8SORDc8YWHrYYc8Uw1P7aAjscV1iyYXgjavgDfRDW7yADusWD6oFS2s1044ciI6/k3JsarGRa+BK3gsutdGvvVQslqSA/BQ0v93eJrbhmfEF9xLXIbloocFL3Abvvg3XxFx8L95uuadKtFrwLwCcsGlyxkmfbDSyBUs4XG/C4etnAn32proEc89UhTMGbZbORt+PfzV96bH4ykyPgE/iF/uozz53wAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAAq0lEQVR4XmNgGAUDAcSAmANdEB0wAnE4EP8H4l9AnAUVAwFFIF4BxHwwhWVQhc+g9E8gtoUqrgPiGCibwRiIHwFxFJQvA8SHgHg+A8SgVQwQ54HBBAYknVCgwgDRIA3EOWhyWEE5EC9nQLgdLwgC4n/ogriALxCfRhfEBUCKW9EFcYFJDBCnEAT6QHyBARKMBEE0EO8BYm50CWwA5IQGdEFcoBuIDdEFhyIAAIFRGGvrucy1AAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAABVUlEQVR4Xu2VPS8FQRSGXwkJcRNfBa3kdig0NP6AiIqC3EZHL5FohEqhEZVOFFRCISQSpf+gEpHoFCqNAu9rhnvMvexq7uwm8yRPdubMbHLm7MwOkEgkikaFjvpnaeiiO/SZvtJ3ekqH7KQioko/0m3a7WOD9BZuEYVmFi5JqbZoo0c+1ulj32hwgPaGAxEYoU9wW2bcxA/hkv+x/5X4gx+QB8HYBp0zsVag6toKq30Bl59y+kSrvKcdvq+DskWHfV+Jr/j2b0zS+X845l7LjXJTQZX4mh3YRWNymrwAd0hO/DMmSviNrsJUXfSjXnXLHl2mm0E8Bi90CUHif3FFz+F+WzGZoDOmr2K3m35T7ugxckwk+6gf9jyuu9cyqdKbIDaFHF/gEvH2eg+9RuOiv8zkDE0ugxZhL6lQHdxMdGBLiW6wWhgsC9O0LwyWhcUwkEgk4vMBwERIUNLMzkIAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAaCAYAAAC6nQw6AAABIUlEQVR4Xu3Sv0tCURjG8TcsUAmCAocIBG1zjP6DGpsamvwPWqNRl/YEpbmxxa3daujXUhAEQYMtDRFNTdGP79M5dW/namrRlA98wHteee97zrlmwwzzNxnBFHJhod9ksIZ7HOMUZXONB8o+LjDrn9XgCcv+eQ5p/7trJnCHUrB+iyNz9dWg1jHrqFtyGwfmms2jGdQS0ciPKIQF0sIrLjHztWQb2I0vfDQajy/6tPCClWBdZ6Um1fiiJtH4nRodmnuJXqZkUcMJns1NOu1rNooti25LSWERe+a2toRirL6Aa0tu9/1WHrCNHdxgE3lc4Qznn/+Ozqfn5/BddBRtc1Mp+pB/FDVQIzXUcYSfTN/RuegSKmgEtYEzhkn7xTT/MW9L5y6c34jsmAAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAkElEQVR4XmNgGAW0AIxALATErEhivEhsOLgOxP+B+AoQqzBANFZAaTgwRBNwA+IzSHyswACIdwKxPLoEMnAB4h1ALIMugQy8gXgVA8RDOAHIfXOAmBtNTBKJz2AGxM+B+AcQPwXilUB8Goh/ISsSA+L9DBDFIE88YoAE0TcgjkVSBw5cfiQ+BwPEOhA9CogDAIEOEq87fQxPAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABWklEQVR4Xu2UvytGURjHH2FQUiKS5B1JMlhNYjDIm0kZJYtBFjtZ2Mwkg4Es8gd4syCz0UDKRCkm+fH5OufNuee+73KvlHq/9al7v8/t9Jzvc+4xq+kvdQl38AIHUJ8sZ9cUHMMnLEe1XKqDfXiDkaiWS+1wDTfQGdVyacGSMTTbL+W8ay6GeTiDe3iGifCjLFIM6vgCCt678rT690x6NZdvIfBKcAtdgVdJ3TAYm2Wp25XIezC3uPKupkY4hL24IGmr8WnQ8fuAscCL1WFuNoqvZBUaGIYjaAg8ba+cr6KYDmplLcG4uW4rzmLW0jHMwLp/XrR0XdowF8WWVZnFtqX/Ni00Cb1wAj3J8vfuBvyzvn2E/p+yy6UEbaGJhuDU3OU0GtUkRaAOxZq5U6VIc2kOisG7BqxBa4eZ1Qc70BR46lQdZ74Vz+Hds+m9VXgy9x/oLm/xfk3/VV+ulj9qvgPalQAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAZCAYAAADe1WXtAAABPklEQVR4Xu2UsUoDQRCGR0whxMKgGASDia9gm0psLFRI5wOIvQQEqzR5g5QBO0GwCWKlkGAn1paCBsFCEISkUVD/P3Ph5iZccQsWQj74OGZm2b3Z3TuRKX/JHezDATx3tWD2YAf+wCNXC2YJPsBHWHS1YDbgEF7AnKsFcyjJ1pfhbFwO4xR+wQN4C9/hB9y2g7LC1rmf5Siehz34DFeiXGbY+rGJ1+Gr6MRcwLIv8aHewzOYT4wAM6KtV01uV3ShpslZuGhabURBJq9SC37DLZOzMJ9WG8Hr5K/Sk2hrXJB7WjM1ciL6tqm0Jdk6Yes7cA1ewpKprcIXE08wPuVFl3+DXdF/wqarse1Pl+O5JD6aBRtEzIm2zaeHrfNmWCqiEwfBRa7gjclxsrqJM3Et+qVxv/kc/yoZN+JhU/4FvwBuOQzclZ0uAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,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>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAjElEQVR4XmNgGAVDHnQC8X8g/gXEj6BsEH4G5ccjlDIwsALxXyDOB2JmqNh8IP4HxB4wRcjAD4gnADEjkpgvA8SGciQxMNAE4rdAzIMmDrIBpAFkGAqwBOKf6IJAcB2KxdEl9IH4E7ogA8RPweiCMCADxHeAeC4QnwPikwyo/sEKBIBYkgHTL6OA9gAAm8EZXYWo1YsAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAZCAYAAAB3oa15AAACZklEQVR4Xu2WTYhOURjHHyFkhCGaRol8ZEYoNspsJimJBRbKUliJLJAmvRsLVhJFlI+SzZSFsGChlBIbyo58pETZsUDh/5vnPr3nPfedBdOruXV/9eu+95z7cc7zPOe816ympia4L9/Lz/Jta1c12CJPyd/yTtZXCSbJYflTbsj6KkGveem8lvNbu6rBRvnLPAtkg0lMbblinHPcvP45XpQf5A95WU5Prhu3fJTf5UDSdtR8UhzbMUHOkbPyjg5CdWAJBnrFfFDAkXPaN8dFCTzkrLxq/2fb3SO/yi9yRdY3AgPdn5zPlS/NS2lR0h6slC/kQfkp6+sUh+UjOSPv6JLf5NqkLRb1TWufMibb9mEdIrZ5sl5isXwqZydt58yzsqs4z9PG4r5hzZLrNOyKbPHb8w44KXdnbWSEAU6UR6w80Hdya3JOmTXkQvMXAX+Ir8wDBOvlPfOsvZGri3bu2VH85n2nzd/He/nEmWke0DzII7BFPpDLs/a75jsTR6KdwwRiALCmcJs1J8DuxbNjG2bCBOaEXGI+SEqDMl1a2Cj6WWMsWJ4Ho5YPsAZyYovEPPrAAmehp8RgiFbULNkNFsjH5qUZWY3S2Ck3mb8PmDwbCPcA1+RVMibiHzslrVM+TfhNBIn2MnlITjPPyEPzwPWYZzPnmrVmj2v65So5JS76V4h8pDaFCRH92/KCfC4vyQPmX7u3zKN4xryug3Vynzwmr8s+88gzATLBM57J83KwuGdMUPuxMNsR/8yTZbc1S5Dz0b6x5lm5lFnQBIv7uCfv/2v2mu8eQ1Yun0rQkE/Mt8KaGvEHgBxtEdmE+1sAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,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>
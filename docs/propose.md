Tôi xin trình bày **Kiến trúc Chi tiết Hệ thống Dự báo Chuyển lượt (Turn-Taking) cho Tiếng Việt trên Thiết bị Biên**.

Chúng ta sẽ gọi hệ thống này là **"Viet-TurnEdge Hybrid"**.

---

### TỔNG QUAN KIẾN TRÚC (HIGH-LEVEL ARCHITECTURE)

Hệ thống hoạt động theo mô hình **Late Fusion (Hợp nhất muộn)** với cơ chế **Soft-Gating**. Hai luồng dữ liệu (Âm thanh & Văn bản) được xử lý song song bởi hai mạng nơ-ron chuyên biệt, sau đó hợp nhất để đưa ra quyết định cuối cùng.

* 
**Mục tiêu:** Độ trễ xử lý < 100ms trên Raspberry Pi 4/5.


* **Input:** Microphone Stream & ASR Text Stream.
* **Output:** 3 trạng thái xác suất: `Turn-Yield` (Ngắt lời), `Turn-Hold` (Giữ lời), `Backchannel` (Phản hồi ậm ừ).

---

### CHI TIẾT TỪNG MODULE

#### 1. Nhánh Âm thanh (Acoustic Stream - "The Fast Lane")

Đây là nhánh chạy nhanh nhất, chịu trách nhiệm bắt tín hiệu ngữ điệu và khoảng lặng.

* **Đầu vào (Input Features):**
* Cửa sổ trượt (Sliding Window): 20ms, chồng lấp (stride) 10ms.
* **Đặc trưng cốt lõi:** Log-Mel Spectrogram (như truyền thống).
* **Đặc trưng bổ sung cho tiếng Việt:**
* 
**Cường độ năng lượng (Intensity/Energy):** Cực kỳ quan trọng vì người Việt dùng cường độ để đánh dấu ngắt câu nhiều hơn cao độ.


* 
**F0 Pitch Contour:** Để phát hiện thanh điệu (Huyền/Nặng thường xuống, Sắc/Hỏi thường lên).






* 
**Mô hình (Backbone): Causal Dilated TCN (Mạng Tích chập Thời gian Nhân quả)**.


* **Cấu trúc:** 4 lớp Residual Block (Khối dư).
* **Kernel Size:** 3.
* 
**Dilation:**  (Tăng vùng nhìn theo cấp số nhân để "nhớ" được ngữ cảnh 2-3 giây trước đó mà không tốn bộ nhớ như GRU).


* 
**Causal Padding:** Đảm bảo chỉ nhìn về quá khứ, không rò rỉ thông tin tương lai.


* 
**Ưu điểm:** Tính toán song song cực nhanh, độ trễ < 5ms.




* **Đầu ra:** Vector đặc trưng âm thanh .

#### 2. Nhánh Ngôn ngữ (Linguistic Stream - "The Smart Lane")

Đây là nhánh "thông minh", chịu trách nhiệm hiểu ngữ nghĩa và hư từ cuối câu.

* **Đầu vào:** Token văn bản từ Streaming ASR (như Whisper Tiny hoặc PhoWhisper).
* 
**Mô hình (Backbone): Distilled PhoBERT (TinyBERT architecture)**.


* 
**Kỹ thuật nén:** Sử dụng **Knowledge Distillation (Chưng cất tri thức)** để nén PhoBERT-base (135M tham số) xuống kích thước TinyBERT (~14M tham số).


* **Tối ưu hóa phần cứng:** Chuyển đổi sang định dạng **ONNX** và **Lượng tử hóa (Quantization) INT8**. Điều này giúp giảm kích thước model 4 lần và tăng tốc độ 3-4 lần.


* 
**Nhiệm vụ:** Phát hiện các "Traffic Signals" (Hư từ): *nhé, nhỉ, à* (Yield) hoặc *mà, thì, là* (Hold).




* **Đầu ra:** Vector đặc trưng ngôn ngữ .

#### 3. Tầng Hợp nhất (Fusion Layer & Decision)

Nơi quyết định dựa trên độ tin cậy của từng nhánh.

* 
**Cơ chế:** **Gated Multimodal Unit (GMU)**.


* Mạng sẽ học một cổng  (giá trị 0-1) dựa trên ngữ cảnh hiện tại.
* Ví dụ: Nếu văn bản nhận diện từ "nhé"   nghiêng về Văn bản (tin cậy cao). Nếu văn bản chưa tới (do ASR trễ) nhưng âm thanh thấy im lặng dài + giảm năng lượng   nghiêng về Âm thanh.




* 
**Công thức:** .


* 
**Classifier:** Một lớp Fully Connected đơn giản + Softmax để ra xác suất 3 lớp (Yield, Hold, Backchannel).



---

### CHIẾN LƯỢC HUẤN LUYỆN (TRAINING STRATEGY)

Để giải quyết vấn đề **không gán nhãn thủ công** và **dữ liệu mất cân bằng**:

#### 1. Dữ liệu & Nhãn (Weak Supervision)

* 
**Nguồn:** Podcast tiếng Việt (Vietcetera, Spiderum...) và VLSP 2020 Spontaneous.


* **Gán nhãn tự động:**
* Dùng **Pipeline**: Diarization (tách người nói)  ASR (có timestamp)  **LLM-as-a-Judge** (GPT-4o/Claude).


* 
**Prompt cho LLM:** Đưa đoạn hội thoại và yêu cầu LLM xác định đâu là điểm ngắt lượt dựa trên văn bản và ngữ cảnh.





#### 2. Hàm mất mát (Loss Function)

* Sử dụng **Focal Loss** () thay vì Cross-Entropy.


* **Lý do:** Sự kiện chuyển lượt (TRP) rất hiếm so với các frame "đang nói" hoặc "im lặng". Focal Loss sẽ phạt nặng nếu model đoán sai các điểm chuyển lượt quan trọng này, giúp model không bị bias về lớp đa số.





#### 3. Kỹ thuật Training (Modality Dropout)

* Áp dụng **Random Modality Dropout (RMDT)**.


* Trong quá trình train, thỉnh thoảng ngẫu nhiên tắt ("dropout") tín hiệu Văn bản (cho về 0).
* 
**Mục đích:** Ép nhánh Âm thanh (TCN) phải học cực tốt các đặc trưng cường độ/ngữ điệu để tự ra quyết định khi ASR bị trễ (latency) hoặc ASR bị sai.





---

### TỔNG KẾT LUỒNG XỬ LÝ (RUNTIME FLOW)

1. **0ms:** Audio frame (20ms) đi vào  Trích xuất Mel + Energy + F0.
2. **+5ms:** **TCN** xử lý xong, ra vector .
3. 
**Song song:** Nếu có text từ ASR  **TinyBERT (INT8)** xử lý (30-40ms) ra vector . Nếu chưa có text, dùng buffer cũ hoặc vector rỗng.


4. **+1ms:** **GMU** hợp nhất  và , tính toán xác suất.
5. **Quyết định:**
* Nếu : Gửi tín hiệu ngắt mic/trả lời ngay.
* Nếu : Gửi file âm thanh ngắn ("ừ", "vâng") mà không ngắt quy trình nghe.





Đây là kiến trúc tối ưu nhất để cân bằng giữa độ chính xác học thuật (Academic SOTA) và tính khả thi kỹ thuật (Engineering Feasibility) cho dự án của bạn.
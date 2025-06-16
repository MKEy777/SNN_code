`timescale 1ns / 1ps

module argmax #(
    parameter DATA_WIDTH = 32
)(
    // ϵͳ�ź�
    input  wire                      clk,                // ʱ���ź�
    input  wire                      rst_n,              // �첽�͵�ƽ��λ�ź�
    
    // �����ź�
    input  wire                      i_valid,            // ������Ч�ź�
    output reg                       i_ready,            // ��������ź�
    input  wire                      o_ready,            // ��������ź�
    output reg                       o_valid,            // �����Ч�ź�
    
    // ���ݶ˿�
    input  wire signed [3*DATA_WIDTH-1:0] i_logits,      // ���� logits ����
    output reg [1:0]                 o_predicted_class   // ���Ԥ�����
);

    // �ڲ��ź�
    reg [1:0] predicted_class_reg;    // �Ĵ�Ԥ�����
    reg       data_processed;         // ���ݴ����־

    // ����߼������� argmax
    wire signed [DATA_WIDTH-1:0] logit_0 = i_logits[1*DATA_WIDTH-1 : 0*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_1 = i_logits[2*DATA_WIDTH-1 : 1*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_2 = i_logits[3*DATA_WIDTH-1 : 2*DATA_WIDTH];
    
    wire [1:0] predicted_class_comb = (logit_0 >= logit_1 && logit_0 >= logit_2) ? 2'd0 :
                                      (logit_1 >  logit_0 && logit_1 >= logit_2) ? 2'd1 :
                                                                            2'd2;

    // ʱ���߼����Ĵ��������������źŹ���
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            i_ready <= 1'b1;          // ��ʼ״̬Ϊ׼���ý�������
            predicted_class_reg <= 2'd0;
            data_processed <= 1'b0;
        end else begin
            if (i_valid && i_ready) begin
                // �����������ݣ����㲢�洢���
                predicted_class_reg <= predicted_class_comb;
                data_processed <= 1'b1;
                i_ready <= 1'b0;      // ��ʾģ��æµ
            end else if (data_processed && o_ready) begin
                // ����׼���ã�������
                o_valid <= 1'b1;
                o_predicted_class <= predicted_class_reg;
                data_processed <= 1'b0;
            end else if (o_valid && o_ready) begin
                // ��������ν��գ���λ״̬
                o_valid <= 1'b0;
                i_ready <= 1'b1;      // ׼��������һ������
            end
        end
    end

endmodule
`timescale 1ns / 1ps
//
// ����: �ҳ���������logits��ֵ����һ����������
//       DATA_WIDTH ����ʹ��������֧�ֲ�ͬλ������롣
//
module argmax #(
    parameter DATA_WIDTH = 16 
)(
    // --- ϵͳ�ź� ---
    input  wire                      clk,
    input  wire                      rst_n,
    
    // --- �����ź� ---
    input  wire                      i_valid,
    output wire                      i_ready,
    input  wire                      o_ready,
    output wire                      o_valid,
    
    // --- ���ݶ˿� ---
    input  wire signed [3*DATA_WIDTH-1:0] i_logits,
    output wire [1:0]                     o_predicted_class
);

    // --- ����߼�����·�� ---
    // ���ݴ����DATA_WIDTH�Զ��з���������
    wire signed [DATA_WIDTH-1:0] logit_0 = i_logits[1*DATA_WIDTH-1 : 0*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_1 = i_logits[2*DATA_WIDTH-1 : 1*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_2 = i_logits[3*DATA_WIDTH-1 : 2*DATA_WIDTH];

    assign o_predicted_class = (logit_0 >= logit_1 && logit_0 >= logit_2) ? 2'd0 :
                               (logit_1 >  logit_0 && logit_1 >= logit_2) ? 2'd1 :
                                                                            2'd2;
                                                                            
    // --- ����߼�����·�� ---
    assign o_valid = i_valid;
    assign i_ready = o_ready;

endmodule
`timescale 1ns / 1ps

module top (
    // =================================================================
    // == ����˿ڶ���
    // =================================================================
    // --- ϵͳ�ź� ---
    input  wire                          clk,
    input  wire                          rst_n,
    // --- �������ֽӿ� ---
    input  wire                          i_valid,             // �ⲿϵͳ������Ч����
    output wire                          i_ready,             // ������׼���ý���������
    output wire                          o_valid,             // �����������Ч���
    input  wire                          o_ready,             // �ⲿϵͳ׼���ý��ս��
    // --- �������ݽӿ� (���޸�) ---
    input  wire signed [300*32-1:0]      i_features,          // �޸�: ������������ֱ��Ϊ300��
    output wire [1:0]                    o_predicted_class    // ���������������� (0, 1, �� 2)
);

    // =================================================================
    // == �ڲ���������
    // =================================================================
    localparam DATA_WIDTH      = 32;
    localparam FRACTIONAL_BITS = 16;

    // --- ����ߴ綨�� (�Ѹ���) ---
    // localparam P0_TOTAL_FEATURES = 682; // ���Ƴ�
    localparam P1_K_FEATURES   = 300; // �����������
    localparam P2_OUTPUT_SIZE  = 256; // ��1���ز��
    localparam P3_OUTPUT_SIZE  = 128; // ��2���ز��
    localparam P4_OUTPUT_SIZE  = 3;   // ������

    // --- ROM �ļ�·������ (�Ѹ���) ---
    localparam P2_WEIGHT_ROM_FILE = "layer_0_weights.mem";
    localparam P2_BIAS_ROM_FILE   = "layer_0_combined_bias.mem";
    localparam P3_WEIGHT_ROM_FILE = "layer_1_weights.mem";
    localparam P3_BIAS_ROM_FILE   = "layer_1_combined_bias.mem";
    localparam P4_WEIGHT_ROM_FILE = "layer_2_weights.mem";
    localparam P4_BIAS_ROM_FILE   = "layer_2_bias.mem";



    // --- ��ˮ���������� ---
    wire signed [P2_OUTPUT_SIZE*DATA_WIDTH-1:0] w_p2_spikes;    // P2 -> P3
    wire signed [P3_OUTPUT_SIZE*DATA_WIDTH-1:0] w_p3_spikes;    // P3 -> P4
    wire signed [P4_OUTPUT_SIZE*DATA_WIDTH-1:0] w_p4_logits;    // P4 -> P5

    // --- ��ˮ�������ź����� ---
    wire w_valid_p2_to_p3, w_ready_p3_to_p2;
    wire w_valid_p3_to_p4, w_ready_p4_to_p3;
    wire w_valid_p4_to_p5, w_ready_p5_to_p4;

    // --- �׶�1: ��һ�����ز� ---
    dense_layer #(
        .INPUT_SIZE           (P1_K_FEATURES),
        .OUTPUT_SIZE          (P2_OUTPUT_SIZE),
        .DATA_WIDTH           (DATA_WIDTH),
        .FRACTIONAL_BITS      (FRACTIONAL_BITS),
        .WEIGHT_ROM_FILE      (P2_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P2_BIAS_ROM_FILE),
        .T_MIN                (32'h00010000), 
        .T_MAX                (32'h00020000)
    ) dense_layer_1_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        // �޸�: ֱ�����ӵ���������
        .i_valid              (i_valid),
        .i_ready              (i_ready),
        .o_valid              (w_valid_p2_to_p3),
        .o_ready              (w_ready_p3_to_p2),
        // �޸�: ֱ��ʹ�ö�������i_features
        .i_spikes             (i_features),
        .o_spikes             (w_p2_spikes)
    );

    // --- �׶�2: �ڶ������ز�  ---
    dense_layer #(
        .INPUT_SIZE           (P2_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P3_OUTPUT_SIZE),
        .DATA_WIDTH           (DATA_WIDTH),
        .FRACTIONAL_BITS      (FRACTIONAL_BITS),
        .WEIGHT_ROM_FILE      (P3_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P3_BIAS_ROM_FILE),
        .T_MIN                (32'h00020000), 
        .T_MAX                (32'h00030000)
    ) dense_layer_2_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_valid              (w_valid_p2_to_p3),
        .i_ready              (w_ready_p3_to_p2),
        .o_valid              (w_valid_p3_to_p4),
        .o_ready              (w_ready_p4_to_p3),
        .i_spikes             (w_p2_spikes),
        .o_spikes             (w_p3_spikes)
    );

    // --- �׶�3: ����� ---
    output_layer #(
        .INPUT_SIZE           (P3_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P4_OUTPUT_SIZE),
        .DATA_WIDTH           (DATA_WIDTH),
        .FRACTIONAL_BITS      (FRACTIONAL_BITS),
        .WEIGHT_ROM_FILE      (P4_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P4_BIAS_ROM_FILE)
    ) output_layer_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_valid              (w_valid_p3_to_p4),
        .i_ready              (w_ready_p4_to_p3),
        .o_valid              (w_valid_p4_to_p5),
        .o_ready              (w_ready_p5_to_p4),
        .i_spikes             (w_p3_spikes),
        .o_logits             (w_p4_logits)
    );

    // --- �׶�4: Argmax���� ---
    argmax #(
        .DATA_WIDTH (DATA_WIDTH)
    ) argmax_inst (
        .clk               (clk),
        .rst_n             (rst_n),
        .i_valid           (w_valid_p4_to_p5),
        .i_ready           (w_ready_p5_to_p4),
        .o_ready           (o_ready),
        .o_valid           (o_valid),
        .i_logits          (w_p4_logits),
        .o_predicted_class (o_predicted_class)
    );

endmodule
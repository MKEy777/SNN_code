// �ļ���: quantization_unit_3stage_final.v
// ģ��: quantization_unit (3����ˮ�����հ�)
// ����: �Ϊ3����ˮ�ߣ����ӷ��ͳ˷����룬�Ի������ʱ��ԣ����
`timescale 1ns / 1ps

module quantization_unit #(
    parameter integer DATA_WIDTH       = 8,
    parameter integer ACC_WIDTH        = 32,
    parameter integer BIAS_WIDTH       = 32,
    parameter signed [31:0] OUTPUT_ZERO_POINT  = 180,
    parameter signed [31:0] OUTPUT_MULTIPLIER  = 1198999149,
    parameter integer       OUTPUT_SHIFT       = 11
)(
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire                                i_valid,
    output wire                                o_valid,
    input  wire signed [ACC_WIDTH-1:0]       accumulator_in,
    input  wire signed [BIAS_WIDTH-1:0]        bias_in,
    output wire signed [DATA_WIDTH-1:0]       clamped_output
);

    // =================================================================
    // == 1. ��ˮ�߼Ĵ�������
    // =================================================================
    // --- ��1 -> ��2 ����ͨ·�Ĵ���: �洢�ӷ���� ---
    reg signed [ACC_WIDTH:0]    acc_with_bias_reg;

    // --- ��2 -> ��3 ����ͨ·�Ĵ���: �洢�˷���� ---
    reg signed [ACC_WIDTH+32:0] mult_reg; // λ��������ƥ��˷������

    // --- ����ͨ·�Ĵ��� (3����ˮ�� -> 3�����ӳ�) ---
    reg [2:0] valid_pipeline_reg;
    assign o_valid = valid_pipeline_reg[2];

    // =================================================================
    // == 2. ����߼�����
    // =================================================================
    // --- ��1 �ļ����߼� (�ӷ�) ---
    wire signed [ACC_WIDTH:0] acc_with_bias_comb = accumulator_in + bias_in;

    // --- ��2 �ļ����߼� (�˷�) ---
    wire signed [ACC_WIDTH+32:0] mult_comb = acc_with_bias_reg * OUTPUT_MULTIPLIER;

    // --- ��3 �ļ����߼� (��λ, �����, ����) ---
    wire signed [ACC_WIDTH+32:0] shifted = $signed(mult_reg) >>> OUTPUT_SHIFT;
    wire signed [ACC_WIDTH+32:0] with_zp;
    wire signed [ACC_WIDTH+32:0] output_zero_point_extended = OUTPUT_ZERO_POINT;
    assign with_zp = shifted + output_zero_point_extended;
    
    localparam signed [DATA_WIDTH-1:0] MAX_VAL = {1'b0, {DATA_WIDTH-1{1'b1}}};
    localparam signed [DATA_WIDTH-1:0] MIN_VAL = {1'b1, {DATA_WIDTH-1{1'b0}}};
    wire is_overflow  = (with_zp > MAX_VAL);
    wire is_underflow = (with_zp < MIN_VAL);
    assign clamped_output = is_overflow  ? MAX_VAL :
                            is_underflow ? MIN_VAL : with_zp[DATA_WIDTH-1:0];

    // =================================================================
    // == 3. ʱ���߼� (����������ˮ�߼Ĵ���)
    // =================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_with_bias_reg  <= 0;
            mult_reg           <= 0;
            valid_pipeline_reg <= 3'b0;
        end else begin
            // --- ������ˮ�� ---
            // ��1 -> ��2: ����ӷ����
            acc_with_bias_reg <= acc_with_bias_comb;
            
            // ��2 -> ��3: ����˷����
            mult_reg <= mult_comb;
            
            // --- ������ˮ�� ---
            valid_pipeline_reg <= {valid_pipeline_reg[1:0], i_valid};
        end
    end

endmodule
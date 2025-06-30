// �ļ���: parallel_compute_engine_pipelined_v2001.v
// ģ��: parallel_compute_engine (3����ˮ��, �ϸ�Verilog-2001��)
// ����: ���м������棬�ڲ�����3����ˮ�ߣ���ʵ�ָ��ߵ�ʱ��Ƶ�ʡ�
`timescale 1ns / 1ps

module parallel_compute_engine #(
    parameter PARALLEL_FACTOR = 8,
    parameter DATA_WIDTH      = 8,
    parameter ACC_WIDTH       = 32,
    parameter INPUT_ZERO_POINT = 0
)(
    // �ӿڶ˿� (Verilog-2001 ����)
    input  wire                                         clk,
    input  wire                                         rst_n,
    input  wire                                         i_valid,
    output wire                                         o_valid,
    input  wire signed [PARALLEL_FACTOR*DATA_WIDTH-1:0] parallel_inputs,
    input  wire signed [PARALLEL_FACTOR*DATA_WIDTH-1:0] parallel_weights,
    output wire signed [ACC_WIDTH-1:0]                  sum_of_products
);

    // =================================================================
    // == 1. �����ͼĴ�������
    // =================================================================
    
    // --- 2001����: ѭ������������ģ�鶥������Ϊ integer ---
    integer k; 

    // --- ��ˮ�����ݼĴ��� ---
    (* use_dsp = "yes" *) 
    reg signed [DATA_WIDTH*2+1:0] products_reg [0:PARALLEL_FACTOR-1];
    reg signed [ACC_WIDTH-1:0] sum_stage1_reg [0:PARALLEL_FACTOR/2-1];
    reg signed [ACC_WIDTH-1:0] sum_of_products_reg;
    
    // --- ������ˮ�߼Ĵ��� ---
    reg [2:0] valid_pipeline_reg;

    // --- ����߼����� ---
    // --- 2001����: generate����ʹ�õ�wire�������ⲿ����Ϊ���� ---
    wire signed [DATA_WIDTH+1:0] sub_result [0:PARALLEL_FACTOR-1];
    wire signed [DATA_WIDTH*2+1:0] products_comb [0:PARALLEL_FACTOR-1];
    wire signed [ACC_WIDTH-1:0] sum_stage1_comb [0:PARALLEL_FACTOR/2-1];
    wire signed [ACC_WIDTH-1:0] sum_stage2_comb [0:PARALLEL_FACTOR/4-1];
    wire signed [ACC_WIDTH-1:0] final_sum_comb;

    // --- ����˿����� ---
    assign sum_of_products = sum_of_products_reg;
    assign o_valid = valid_pipeline_reg[2];


    // =================================================================
    // == 2. ����߼�����
    // =================================================================
    genvar i, s1, s2; // generateѭ������
    generate
        // --- ��1 �ļ����߼� (���� -> �˷�) ---
        for (i = 0; i < PARALLEL_FACTOR; i = i + 1) begin : parallel_mac_units
            assign sub_result[i] = $signed(parallel_inputs[i*DATA_WIDTH +: DATA_WIDTH]) - $signed(INPUT_ZERO_POINT);
            assign products_comb[i] = sub_result[i] * $signed(parallel_weights[i*DATA_WIDTH +: DATA_WIDTH]);
        end

        // --- ��2 �ļ����߼� (��һ���ӷ�: 8 -> 4) ---
        for (s1 = 0; s1 < PARALLEL_FACTOR/2; s1 = s1 + 1) begin
            assign sum_stage1_comb[s1] = $signed(products_reg[2*s1]) + $signed(products_reg[2*s1+1]);
        end

        // --- ��3 �ļ����߼� (�ڶ��������ӷ�: 4 -> 2 -> 1) ---
        for (s2 = 0; s2 < PARALLEL_FACTOR/4; s2 = s2 + 1) begin
            assign sum_stage2_comb[s2] = $signed(sum_stage1_reg[2*s2]) + $signed(sum_stage1_reg[2*s2+1]);
        end
    endgenerate
    assign final_sum_comb = $signed(sum_stage2_comb[0]) + $signed(sum_stage2_comb[1]);


    // =================================================================
    // == 3. ʱ���߼� (����������ˮ�߼Ĵ���)
    // =================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ��λ���мĴ���
            // --- 2001����: ʹ��Ԥ�������� integer 'k' ---
            for (k = 0; k < PARALLEL_FACTOR; k = k + 1) begin
                products_reg[k] <= 0;
            end
            for (k = 0; k < PARALLEL_FACTOR/2; k = k + 1) begin
                sum_stage1_reg[k] <= 0;
            end
            sum_of_products_reg <= 0;
            valid_pipeline_reg  <= 3'b0;
        end else begin
            // --- ������ˮ�� ---
            // ��1 -> ��2
            // --- 2001����: ʹ��Ԥ�������� integer 'k' ---
            for (k = 0; k < PARALLEL_FACTOR; k = k + 1) begin
                products_reg[k] <= products_comb[k];
            end

            // ��2 -> ��3
            // --- 2001����: ʹ��Ԥ�������� integer 'k' ---
            for (k = 0; k < PARALLEL_FACTOR/2; k = k + 1) begin
                sum_stage1_reg[k] <= sum_stage1_comb[k];
            end
            
            // ��3 -> ���
            sum_of_products_reg <= final_sum_comb;
            
            // --- ������ˮ�� ---
            valid_pipeline_reg <= {valid_pipeline_reg[1:0], i_valid};
        end
    end

endmodule
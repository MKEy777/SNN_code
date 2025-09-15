`timescale 1ns / 1ps

module cnn_top_streaming #(
    // --- 结构参数 ---
    parameter IMG_H        = 8,
    parameter IMG_W        = 9,
    parameter IMG_C        = 4,
    parameter CONV1_OUT_C  = 8,
    parameter CONV2_OUT_C  = 16,
    parameter CONV2_OUT_H  = 4,
    parameter CONV2_OUT_W  = 5,
    parameter FLAT_LEN     = 320,
    parameter K_DIM        = 3,
    parameter DATA_W       = 8,
    parameter ACC_W        = 32,
    parameter TIME_W       = 32
) (
    // --- 系统接口 ---
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      i_start_inference,

    // --- BRAM 接口 (用于输入图像) ---
    input  wire                      i_bram_we,
    input  wire [$clog2(IMG_H*IMG_W*IMG_C)-1:0] i_bram_addr,
    input  wire signed [DATA_W-1:0]  i_bram_wdata,

    // --- 脉冲输出接口 ---
    output wire                      o_spike_valid,
    input  wire                      i_spike_ack,
    output wire signed [TIME_W-1:0]  o_spike_time,
    output wire [$clog2(FLAT_LEN)-1:0] o_spike_addr,
    output wire                      o_cnn_done
);

//================================================================
// 1. 内部参数定义与ROM实例化 (Modified Section)
//================================================================
    // --- 内部连线 ---
    wire signed [IMG_C*K_DIM*K_DIM*DATA_W-1:0]           i_rom_c1_dw_kernels_flat;
    wire signed [CONV1_OUT_C*IMG_C*DATA_W-1:0]           i_rom_c1_pw_weights_flat;
    wire signed [CONV1_OUT_C*ACC_W-1:0]                  i_rom_c1_biases_flat;
    wire signed [CONV1_OUT_C*32-1:0]                     i_rom_c1_scales_flat;
    wire signed [CONV1_OUT_C*DATA_W-1:0]                 i_rom_c1_zps_flat;
    wire signed [CONV1_OUT_C*K_DIM*K_DIM*DATA_W-1:0]     i_rom_c2_dw_kernels_flat;
    wire signed [CONV2_OUT_C*CONV1_OUT_C*DATA_W-1:0]     i_rom_c2_pw_weights_flat;
    wire signed [CONV2_OUT_C*ACC_W-1:0]                  i_rom_c2_biases_flat;
    wire signed [CONV2_OUT_C*32-1:0]                     i_rom_c2_scales_flat;
    wire signed [CONV2_OUT_C*DATA_W-1:0]                 i_rom_c2_zps_flat;

    // --- 定义常量 ---
    localparam [15:0] CONV1_ZERO_POINT = 16'h000B;
    localparam [15:0] CONV2_ZERO_POINT = 16'h0008;
    localparam signed [31:0] CONV1_BIAS_CH0 = 32'hFFFFFFF2, CONV1_BIAS_CH1 = 32'h00000041, CONV1_BIAS_CH2 = 32'hFFFFFFF9, CONV1_BIAS_CH3 = 32'h00000008, CONV1_BIAS_CH4 = 32'hFFFFFCC3, CONV1_BIAS_CH5 = 32'hFFFFFCC0, CONV1_BIAS_CH6 = 32'h00000005, CONV1_BIAS_CH7 = 32'hFFFFFFF0;
    localparam signed [31:0] CONV2_BIAS_CH0  = 32'h000000DF, CONV2_BIAS_CH1  = 32'hFFFFFFF8, CONV2_BIAS_CH2  = 32'h0000005E, CONV2_BIAS_CH3  = 32'h00000036, CONV2_BIAS_CH4  = 32'h00000080, CONV2_BIAS_CH5  = 32'h00000069, CONV2_BIAS_CH6  = 32'h00000031, CONV2_BIAS_CH7  = 32'h00000007, CONV2_BIAS_CH8  = 32'hFFFFFF8D, CONV2_BIAS_CH9  = 32'h0000005E, CONV2_BIAS_CH10 = 32'h00000030, CONV2_BIAS_CH11 = 32'h00000006, CONV2_BIAS_CH12 = 32'hFFFFFFD7, CONV2_BIAS_CH13 = 32'h00000015, CONV2_BIAS_CH14 = 32'h0000003B, CONV2_BIAS_CH15 = 32'h00000031;

    localparam C1_DW_KERNEL_W = K_DIM*K_DIM*DATA_W; 
    wire signed [C1_DW_KERNEL_W-1:0] c1_dw_k_ch0, c1_dw_k_ch1, c1_dw_k_ch2, c1_dw_k_ch3;
    
    // Instantiate a ROM for each channel's kernel file
    rom_sync #(.DATA_WIDTH(C1_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv1_dw_kernel_ch0.mem"))
        Inst_C1_DW_K_ROM_CH0 (.clk(clk), .addr(1'b0), .data_out(c1_dw_k_ch0));
    
    rom_sync #(.DATA_WIDTH(C1_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv1_dw_kernel_ch1.mem"))
        Inst_C1_DW_K_ROM_CH1 (.clk(clk), .addr(1'b0), .data_out(c1_dw_k_ch1));
    
    rom_sync #(.DATA_WIDTH(C1_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv1_dw_kernel_ch2.mem"))
        Inst_C1_DW_K_ROM_CH2 (.clk(clk), .addr(1'b0), .data_out(c1_dw_k_ch2));
    
    rom_sync #(.DATA_WIDTH(C1_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv1_dw_kernel_ch3.mem"))
        Inst_C1_DW_K_ROM_CH3 (.clk(clk), .addr(1'b0), .data_out(c1_dw_k_ch3));
    
    // Concatenate individual kernels back into the flat wire the design expects
    assign i_rom_c1_dw_kernels_flat = {c1_dw_k_ch3, c1_dw_k_ch2, c1_dw_k_ch1, c1_dw_k_ch0};
    
    rom_sync #(.DATA_WIDTH(CONV1_OUT_C*IMG_C*DATA_W), .ADDR_WIDTH(1), .MEM_FILE("conv1_pw_weights.mem")) 
        Inst_C1_PW_Weight_ROM (.clk(clk), .addr(1'b0), .data_out(i_rom_c1_pw_weights_flat));
    
    rom_sync #(.DATA_WIDTH(CONV1_OUT_C*32), .ADDR_WIDTH(1), .MEM_FILE("conv1_scale.mem")) 
        Inst_C1_Scale_ROM (.clk(clk), .addr(1'b0), .data_out(i_rom_c1_scales_flat));

    localparam C2_DW_KERNEL_W = K_DIM*K_DIM*DATA_W; 
    wire signed [C2_DW_KERNEL_W-1:0] c2_dw_kernels [0:CONV1_OUT_C-1];
    genvar i;
    generate
        for (i = 0; i < CONV1_OUT_C; i = i + 1) begin : C2_DW_ROMS
            case(i)
                0: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch0.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                1: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch1.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                2: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch2.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                3: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch3.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                4: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch4.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                5: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch5.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                6: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch6.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
                7: rom_sync #(.DATA_WIDTH(C2_DW_KERNEL_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_dw_kernel_ch7.mem"))
                    Inst_C2_DW_K_ROM_CH (.clk(clk), .addr(1'b0), .data_out(c2_dw_kernels[i]));
            endcase
        end
    endgenerate
    
    generate
        for (i = 0; i < CONV1_OUT_C; i = i + 1) begin : pack_c2_kernels
            assign i_rom_c2_dw_kernels_flat[(i+1)*C2_DW_KERNEL_W-1 -: C2_DW_KERNEL_W] = c2_dw_kernels[i];
        end
    endgenerate

    rom_sync #(.DATA_WIDTH(CONV2_OUT_C*CONV1_OUT_C*DATA_W), .ADDR_WIDTH(1), .MEM_FILE("conv2_pw_weights.mem")) 
        Inst_C2_PW_Weight_ROM (.clk(clk), .addr(1'b0), .data_out(i_rom_c2_pw_weights_flat));
    
    rom_sync #(.DATA_WIDTH(CONV2_OUT_C*32), .ADDR_WIDTH(1), .MEM_FILE("conv2_scale.mem")) 
        Inst_C2_Scale_ROM (.clk(clk), .addr(1'b0), .data_out(i_rom_c2_scales_flat));

    assign i_rom_c1_biases_flat = {CONV1_BIAS_CH7, CONV1_BIAS_CH6, CONV1_BIAS_CH5, CONV1_BIAS_CH4, CONV1_BIAS_CH3, CONV1_BIAS_CH2, CONV1_BIAS_CH1, CONV1_BIAS_CH0};
    assign i_rom_c1_zps_flat    = {CONV1_OUT_C{CONV1_ZERO_POINT[DATA_W-1:0]}};
    assign i_rom_c2_biases_flat = {CONV2_BIAS_CH15, CONV2_BIAS_CH14, CONV2_BIAS_CH13, CONV2_BIAS_CH12, CONV2_BIAS_CH11, CONV2_BIAS_CH10, CONV2_BIAS_CH9,  CONV2_BIAS_CH8, CONV2_BIAS_CH7,  CONV2_BIAS_CH6,  CONV2_BIAS_CH5,  CONV2_BIAS_CH4, CONV2_BIAS_CH3,  CONV2_BIAS_CH2,  CONV2_BIAS_CH1,  CONV2_BIAS_CH0};
    assign i_rom_c2_zps_flat    = {CONV2_OUT_C{CONV2_ZERO_POINT[DATA_W-1:0]}};

    localparam S_IDLE       = 2'd0;
    localparam S_STREAMING  = 2'd1;
    localparam S_WAIT_DONE  = 2'd2; 
    reg [1:0] state;

    reg [$clog2(IMG_H*IMG_W)-1:0] pixel_stream_cnt;
    wire pipeline_en = (state == S_STREAMING) || (state == S_WAIT_DONE);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            case(state)
                S_IDLE:
                    if (i_start_inference) state <= S_STREAMING;
                S_STREAMING:
                    if (pixel_stream_cnt == IMG_H*IMG_W - 1'b1) state <= S_WAIT_DONE;
                S_WAIT_DONE:
                    if (o_cnn_done) state <= S_IDLE;
                default:
                    state <= S_IDLE;
            endcase
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            pixel_stream_cnt <= 0;
        end else if (state == S_IDLE) begin
             pixel_stream_cnt <= 0;
        end else if (state == S_STREAMING) begin
            if (pixel_stream_cnt < IMG_H*IMG_W - 1'b1) begin
                pixel_stream_cnt <= pixel_stream_cnt + 1;
            end
        end
    end

//================================================================
// 3. Input Memory (BRAM)
//================================================================
    reg signed [DATA_W-1:0] input_bram[0:IMG_H*IMG_W*IMG_C-1];
    always @(posedge clk) begin
        if (i_bram_we) begin
            input_bram[i_bram_addr] <= i_bram_wdata;
        end
    end

//================================================================
// 4. CONV1 & CONV2 流水线
//================================================================
    genvar ic1, oc1;
    wire signed [ACC_W-1:0] c1_dw_acc_out [0:IMG_C-1];
    wire c1_dw_acc_valid;
    wire signed [CONV1_OUT_C*ACC_W-1:0] c1_pw_vec_out;
    wire c1_pw_vec_valid;
    wire signed [DATA_W-1:0] c1_requant_out [0:CONV1_OUT_C-1];
    wire c1_requant_valid;
    wire [IMG_C-1:0] c1_dw_acc_valid_all;
    wire [CONV1_OUT_C-1:0] c1_requant_valid_all;

    generate 
    for(ic1=0; ic1<IMG_C; ic1=ic1+1) begin: C1_DW_PATH
        wire signed [K_DIM*K_DIM*DATA_W-1:0] c1_dw_win;
        wire c1_dw_win_valid;
        line_buffer_3x3 #(.DATA_W(DATA_W), .IMG_W(IMG_W))
        c1_line_buffer (.clk(clk), .rst_n(rst_n), .i_en(pipeline_en), .i_data((state == S_STREAMING) ? input_bram[ic1*(IMG_H*IMG_W) + pixel_stream_cnt] : 8'd0), .o_win_flat(c1_dw_win), .o_valid(c1_dw_win_valid));
        depthwise_conv_unit #(.DATA_W(DATA_W))
        c1_dw_conv_unit (.clk(clk), .rst_n(rst_n), .i_valid(c1_dw_win_valid), .i_win_flat(c1_dw_win), .i_kernel_flat(i_rom_c1_dw_kernels_flat[(ic1+1)*(K_DIM*K_DIM*DATA_W)-1 -: (K_DIM*K_DIM*DATA_W)]), .o_acc(c1_dw_acc_out[ic1]), .o_valid(c1_dw_acc_valid_all[ic1]));
    end
    endgenerate
    assign c1_dw_acc_valid = c1_dw_acc_valid_all[0];

    pointwise_conv_unit #(.DATA_W(DATA_W), .ACC_W(ACC_W), .IN_CH(IMG_C), .OUT_CH(CONV1_OUT_C), .ACC_REG_W(48))
    c1_pw_conv_unit (.clk(clk), .rst_n(rst_n), .i_valid(c1_dw_acc_valid), .i_vec_flat({c1_dw_acc_out[3],c1_dw_acc_out[2],c1_dw_acc_out[1],c1_dw_acc_out[0]}), .i_weights_flat(i_rom_c1_pw_weights_flat), .o_vec_flat(c1_pw_vec_out), .o_valid(c1_pw_vec_valid));
    
    generate
    for(oc1=0; oc1<CONV1_OUT_C; oc1=oc1+1) begin: C1_REQUANT_PATH
        requantize_relu #(.IN_W(ACC_W), .OUT_W(DATA_W))
        c1_requant_unit(.clk(clk), .rst_n(rst_n), .i_valid(c1_pw_vec_valid), .i_acc(c1_pw_vec_out[(oc1+1)*ACC_W-1 -: ACC_W]), .i_bias(i_rom_c1_biases_flat[(oc1+1)*ACC_W-1 -: ACC_W]), .i_scale(i_rom_c1_scales_flat[(oc1+1)*32-1 -: 32]), .i_zero_point(i_rom_c1_zps_flat[(oc1+1)*DATA_W-1 -: DATA_W]), .o_data(c1_requant_out[oc1]), .o_valid(c1_requant_valid_all[oc1]));
    end
    endgenerate
    assign c1_requant_valid = c1_requant_valid_all[0];
    
    // -- Downsampling --
    reg [$clog2(IMG_W)-1:0] c1_output_col_cnt;
    reg [$clog2(IMG_H)-1:0] c1_output_row_cnt;
    wire c2_pipeline_en = c1_requant_valid && (c1_output_col_cnt[0] == 0) && (c1_output_row_cnt[0] == 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c1_output_col_cnt <= 0;
            c1_output_row_cnt <= 0;
        end else if (state == S_IDLE) begin
            c1_output_col_cnt <= 0;
            c1_output_row_cnt <= 0;
        end else if (c1_requant_valid) begin
            if (c1_output_col_cnt == IMG_W - 1) begin
                c1_output_col_cnt <= 0;
                if (c1_output_row_cnt == IMG_H - 1) begin
                    c1_output_row_cnt <= 0;
                end else begin
                    c1_output_row_cnt <= c1_output_row_cnt + 1;
                end
            end else begin
                c1_output_col_cnt <= c1_output_col_cnt + 1;
            end
        end
    end
    
    // -- CONV2 --
    genvar ic2, oc2;
    wire signed [ACC_W-1:0] c2_dw_acc_out [0:CONV1_OUT_C-1];
    wire c2_dw_acc_valid;
    wire signed [CONV2_OUT_C*ACC_W-1:0] c2_pw_vec_out;
    wire c2_pw_vec_valid;
    wire signed [DATA_W-1:0] c2_requant_out [0:CONV2_OUT_C-1];
    wire c2_requant_valid;
    wire [CONV1_OUT_C-1:0] c2_dw_acc_valid_all;
    wire [CONV2_OUT_C-1:0] c2_requant_valid_all;
    wire encoder_is_busy;

    generate
    for(ic2=0; ic2<CONV1_OUT_C; ic2=ic2+1) begin: C2_DW_PATH
        wire signed [K_DIM*K_DIM*DATA_W-1:0] c2_dw_win;
        wire c2_dw_win_valid;
        line_buffer_3x3 #(.DATA_W(DATA_W), .IMG_W(IMG_W/2 + 1))
        c2_line_buffer (.clk(clk), .rst_n(rst_n), .i_en(c2_pipeline_en), .i_data(c1_requant_out[ic2]), .o_win_flat(c2_dw_win), .o_valid(c2_dw_win_valid));
        depthwise_conv_unit #(.DATA_W(DATA_W))
        c2_dw_conv_unit (.clk(clk), .rst_n(rst_n), .i_valid(c2_dw_win_valid), .i_win_flat(c2_dw_win), .i_kernel_flat(i_rom_c2_dw_kernels_flat[(ic2+1)*(K_DIM*K_DIM*DATA_W)-1 -: (K_DIM*K_DIM*DATA_W)]), .o_acc(c2_dw_acc_out[ic2]), .o_valid(c2_dw_acc_valid_all[ic2]));
    end
    endgenerate
    assign c2_dw_acc_valid = c2_dw_acc_valid_all[0];

    pointwise_conv_unit #(.DATA_W(DATA_W), .ACC_W(ACC_W), .IN_CH(CONV1_OUT_C), .OUT_CH(CONV2_OUT_C), .ACC_REG_W(48))
    c2_pw_conv_unit (.clk(clk), .rst_n(rst_n), .i_valid(c2_dw_acc_valid), .i_vec_flat({c2_dw_acc_out[7], c2_dw_acc_out[6], c2_dw_acc_out[5], c2_dw_acc_out[4], c2_dw_acc_out[3], c2_dw_acc_out[2], c2_dw_acc_out[1], c2_dw_acc_out[0]}), .i_weights_flat(i_rom_c2_pw_weights_flat), .o_vec_flat(c2_pw_vec_out), .o_valid(c2_pw_vec_valid));

    generate
    for(oc2=0; oc2<CONV2_OUT_C; oc2=oc2+1) begin: C2_REQUANT_PATH
        requantize_relu #(.IN_W(ACC_W), .OUT_W(DATA_W))
        c2_requant_unit(.clk(clk), .rst_n(rst_n), .i_valid(c2_pw_vec_valid), .i_acc(c2_pw_vec_out[(oc2+1)*ACC_W-1 -: ACC_W]), .i_bias(i_rom_c2_biases_flat[(oc2+1)*ACC_W-1 -: ACC_W]), .i_scale(i_rom_c2_scales_flat[(oc2+1)*32-1 -: 32]), .i_zero_point(i_rom_c2_zps_flat[(oc2+1)*DATA_W-1 -: DATA_W]), .o_data(c2_requant_out[oc2]), .o_valid(c2_requant_valid_all[oc2]));
    end
    endgenerate
    assign c2_requant_valid = c2_requant_valid_all[0];
    
//================================================================
// 5. Streaming Serial Encoder Stage
//================================================================
    AnnToSnnEncoder_serial #(
        .VEC_LEN(FLAT_LEN), .PIXEL_VEC_LEN(CONV2_OUT_C), .NUM_PIXELS(CONV2_OUT_H*CONV2_OUT_W)
    ) i_encoder (
        .clk(clk), .rst_n(rst_n),
        .i_pixel_valid(c2_requant_valid),
        .i_pixel_vec({c2_requant_out[15], c2_requant_out[14], c2_requant_out[13], c2_requant_out[12], c2_requant_out[11], c2_requant_out[10], c2_requant_out[9], c2_requant_out[8], c2_requant_out[7], c2_requant_out[6], c2_requant_out[5], c2_requant_out[4], c2_requant_out[3], c2_requant_out[2], c2_requant_out[1], c2_requant_out[0]}),
        .o_spike_valid(o_spike_valid),
        .i_spike_ack(i_spike_ack),
        .o_spike_time(o_spike_time),
        .o_spike_addr(o_spike_addr),
        .o_busy(encoder_is_busy),
        .o_last_pixel_sent(o_cnn_done)
    );

endmodule
`timescale 1ns / 1ps

/**
 * @brief ��׼ Click ��Ԫ������һ�����ڼ�ص� fire ����ź�
 * @details
 * - �����߼�ʹ�õ�ƽ��������������ȫ�����ۺ�����ë�̷��ա�
 * - `fire` �ź���һ��"�۲촰"������ʾ��״̬�����ı����������������
 * ��������״̬���£��Ӷ���֤����ƵĽ�׳�ԡ�
 */
module click_standard_with_fire_monitor (
    // ����˿�
    input  wire in_req,    // ���������������ź�
    input  wire out_ack,   // ����������Ӧ���ź�
    input  wire rst_n,     // �첽��λ���͵�ƽ��Ч

    // ����˿�
    output wire out_req,   // �����η��������ź�
    output wire in_ack,    // �����η���Ӧ���ź�
    output wire fire       // ���ڼ�صĴ����ź�
);

    // �ڲ�״̬�Ĵ���
    reg s;

    always @(*) begin
        if (!rst_n) begin
            s = 1'b0;
        end else begin
            if (s == out_ack) begin
                s = in_req;
            end else begin
                s = s; 
            end
        end
    end

    assign out_req = s;
    assign in_ack  = s;
    assign fire = (s == out_ack) & (s != in_req);

endmodule
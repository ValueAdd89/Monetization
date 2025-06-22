output "api_url" {
  value = aws_lb.api_lb.dns_name
}
